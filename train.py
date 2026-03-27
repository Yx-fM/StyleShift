#!/usr/bin/env python3
"""Main training script for StyleShift Decoder."""

import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from style_shift.models.vgg import VGG19Encoder
from style_shift.models.adain import AdaIN
from style_shift.models.decoder import Decoder
from style_shift.models.loss import ContentLoss, StyleLoss
from config import TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train StyleShift Decoder")
    
    parser.add_argument("--config", type=str, default="config/default_training.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    return args


def load_checkpoint(path, decoder, optimizer):
    """Load checkpoint from file."""
    checkpoint = torch.load(path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resumed from epoch {epoch}")
    return epoch


def save_checkpoint(path, epoch, decoder, optimizer, config):
    """Save checkpoint to file."""
    checkpoint = {
        'epoch': epoch,
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def main():
    """Main training loop."""
    args = parse_args()
    
    # Load configuration
    config = TrainingConfig.from_yaml(args.config)
    
    # Override config with command line args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    print("="*70)
    print("StyleShift Decoder Training")
    print("="*70)
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Style weight: {config.style_weight}")
    print(f"Content weight: {config.content_weight}")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    print("Loading models...")
    vgg = VGG19Encoder(pretrained=True).to(device)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False  # Freeze VGG
    
    decoder = Decoder().to(device)
    decoder.train()
    
    adain = AdaIN().to(device)
    
    # Loss functions
    content_loss_fn = ContentLoss(weight=config.content_weight)
    style_loss_fn = StyleLoss(weight=config.style_weight)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=config.learning_rate,
        betas=config.betas
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, decoder, optimizer)
    
    # TensorBoard
    log_dir = Path(config.log_dir) / datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard logs: {log_dir}")
    
    # Dummy dataset for now (will be replaced with real dataset)
    # For initial testing
    print("WARNING: Using dummy dataset for initial test")
    print("Replace with real MS-COCO + WikiArt datasets for production")
    
    # Load dataset
    print("Loading dataset...")
    from datasets import StyleTransferDataset
    from torch.utils.data import DataLoader
    
    dataset = StyleTransferDataset(
        content_root=str(config.content_root),
        style_root=str(config.style_root),
        image_size=config.image_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    print(f"  Content images: {len(dataset.content_images)}")
    print(f"  Style images: {len(dataset.style_images)}")
    
    # Training loop
    total_steps = 0
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Reset epoch losses
        loss_total = 0
        loss_style = 0
        num_batches = 0
        
        # Decay learning rate
        lr = config.learning_rate / (1 + config.lr_decay * total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training loop over dataset
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, (content_imgs, style_imgs) in pbar:
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)
            
            # Forward pass
            with torch.no_grad():
                content_feat, _ = vgg(content_imgs)
                _, style_feats = vgg(style_imgs)
                style_feat = style_feats.get('conv4_2', list(style_feats.values())[-1])
            
            adain_feat = adain(content_feat, style_feat)
            stylized = decoder(adain_feat)
            
            # Style loss only (simpler for initial training)
            s_loss = style_loss_fn(stylized, style_imgs)
            loss = s_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update stats
            loss_total += loss.item()
            loss_style += s_loss.item()
            num_batches += 1
            
            total_steps += 1
            
            # Progress bar
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # Logging
            if total_steps % config.log_every == 0:
                writer.add_scalar('Loss/Total', loss.item(), total_steps)
                writer.add_scalar('Loss/Style', s_loss.item(), total_steps)
                writer.add_scalar('Learning Rate', lr, total_steps)
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                Path(config.save_dir) / f"decoder_epoch_{epoch+1}.pth",
                epoch + 1,
                decoder,
                optimizer,
                config
            )
        
        # Print epoch summary
        avg_loss = loss_total / num_batches
        avg_style = loss_style / num_batches
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Style Loss: {avg_style:.4f}")
    
    # Final save
    save_checkpoint(
        Path(config.save_dir) / "decoder_final.pth",
        config.epochs,
        decoder,
        optimizer,
        config
    )
    
    writer.close()
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
