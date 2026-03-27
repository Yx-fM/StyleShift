#!/usr/bin/env python3
"""Download pre-trained StyleShift models."""

import argparse
import hashlib
import os
import requests
from pathlib import Path
from tqdm import tqdm


# Pre-trained model URLs
# Note: These are example URLs - replace with actual model hosting
PRETRAINED_MODELS = {
    'decoder': {
        'url': 'https://example.com/models/styleshift_decoder.pth',
        'hash': 'abc123...',  # SHA256 hash
        'size_mb': 15,
        'description': 'StyleShift Decoder (trained on MS-COCO + WikiArt)',
    },
    'decoder_anime': {
        'url': 'https://example.com/models/styleshift_decoder_anime.pth',
        'hash': 'def456...',
        'size_mb': 15,
        'description': 'StyleShift Decoder - Anime style',
    },
}


def download_file(url, dest, chunk_size=8192):
    """Download file with progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        desc=dest.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def verify_checksum(file_path, expected_hash):
    """Verify file integrity using SHA256."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_hash


def download_model(model_name='decoder', cache_dir=None, force=False):
    """Download pre-trained model."""
    if model_name not in PRETRAINED_MODELS:
        available = ', '.join(PRETRAINED_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    model_info = PRETRAINED_MODELS[model_name]
    
    if cache_dir is None:
        cache_dir = Path.home() / '.cache' / 'styleshift' / 'models'
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / f"{model_name}.pth"
    
    # Check if already downloaded
    if model_path.exists() and not force:
        print(f"Model already exists: {model_path}")
        return model_path
    
    # Download model
    print(f"Downloading {model_name}...")
    print(f"  URL: {model_info['url']}")
    print(f"  Size: {model_info['size_mb']} MB")
    
    try:
        download_file(model_info['url'], model_path)
        
        # Verify checksum
        if model_info.get('hash'):
            print("Verifying checksum...")
            if not verify_checksum(model_path, model_info['hash']):
                model_path.unlink()
                raise RuntimeError("Checksum verification failed!")
            print("  Checksum OK")
        
        print(f"Downloaded to: {model_path}")
        return model_path
    
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Download failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained StyleShift models")
    
    parser.add_argument('model', type=str, nargs='?', default='decoder',
                        help=f"Model name: {list(PRETRAINED_MODELS.keys())}")
    parser.add_argument('--cache-dir', type=str, default=None,
                        help="Cache directory")
    parser.add_argument('--force', action='store_true',
                        help="Force re-download")
    parser.add_argument('--list', action='store_true',
                        help="List available models")
    
    args = parser.parse_args()
    
    print("="*70)
    print("StyleShift Pre-trained Model Downloader")
    print("="*70)
    print()
    
    if args.list:
        print("Available models:")
        for name, info in PRETRAINED_MODELS.items():
            print(f"  {name}:")
            print(f"    Description: {info['description']}")
            print(f"    Size: {info['size_mb']} MB")
        return
    
    try:
        model_path = download_model(args.model, args.cache_dir, args.force)
        print()
        print(f"Model ready: {model_path}")
    except Exception as e:
        print()
        print(f"Error: {e}")
        print()
        print("Note: Pre-trained model URLs are placeholders.")
        print("Please train your own model using:")
        print("  python train.py --epochs 16")


if __name__ == "__main__":
    main()
