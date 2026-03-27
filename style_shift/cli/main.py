#!/usr/bin/env python3
"""Command-line interface for StyleShift."""

import argparse
import sys
from pathlib import Path
from PIL import Image

from style_shift.core.style_transfer import StyleTransfer, StyleTransferConfig
from style_shift.core.config import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StyleShift - Neural Style Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -c photo.jpg -s anime.jpg -o output.jpg
  %(prog)s -c photo.jpg --style-name vangogh --alpha 0.8
  %(prog)s -c photo.jpg -s style.jpg --size 1024 --preserve-color
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-c", "--content",
        type=str,
        required=True,
        help="Content image path"
    )
    
    # Style arguments (mutually exclusive group)
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument(
        "-s", "--style",
        type=str,
        help="Style image path"
    )
    style_group.add_argument(
        "--style-name",
        type=str,
        choices=[
            'anime', 'vangogh', 'monet', 'ukiyoe',
            'mosaic', 'sketch', 'watercolor'
        ],
        help="Built-in style name"
    )
    
    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output image path (default: not saved)"
    )
    
    # Style transfer parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Style strength (0.0-1.0, default: 1.0)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Max output dimension (default: 512)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device selection (default: auto)"
    )
    
    parser.add_argument(
        "--preserve-color",
        action="store_true",
        help="Preserve original colors"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file path"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    # Check content file exists
    if not Path(args.content).exists():
        print(f"Error: Content file not found: {args.content}", file=sys.stderr)
        sys.exit(1)
    
    # Check style file exists if provided
    if args.style and not Path(args.style).exists():
        print(f"Error: Style file not found: {args.style}", file=sys.stderr)
        sys.exit(1)
    
    # Validate alpha
    if not 0.0 <= args.alpha <= 1.0:
        print(f"Error: Alpha must be between 0.0 and 1.0, got {args.alpha}", file=sys.stderr)
        sys.exit(1)
    
    # Validate size
    if args.size <= 0:
        print(f"Error: Size must be positive, got {args.size}", file=sys.stderr)
        sys.exit(1)
    
    # Check that either style or style-name is provided
    if not args.style and not args.style_name:
        print("Error: Must provide either --style or --style-name", file=sys.stderr)
        sys.exit(1)


def main():
    """CLI entry point."""
    args = parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = StyleTransferConfig(
            alpha=args.alpha,
            max_size=args.size,
            device=args.device,
            preserve_color=args.preserve_color
        )
    
    # Initialize StyleTransfer
    print(f"Initializing StyleTransfer (device: {args.device or 'auto'})...")
    st = StyleTransfer(config)
    
    # Perform style transfer
    print(f"Processing: {args.content}")
    print(f"Style: {args.style or args.style_name}")
    print(f"Alpha: {args.alpha}, Size: {args.size}")
    
    result = st.transfer(
        content=args.content,
        style=args.style,
        style_name=args.style_name,
        output_path=args.output,
        alpha=args.alpha
    )
    
    # Output
    if args.output:
        print(f"Saved to: {args.output}")
    else:
        print("Style transfer complete (use --output to save)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
