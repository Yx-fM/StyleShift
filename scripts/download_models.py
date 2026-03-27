#!/usr/bin/env python3
"""Download pre-trained StyleShift models."""

import hashlib
import argparse
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

# Model URLs (replace with actual URLs when available)
MODEL_URL = "https://example.com/models/style_shift_decoder.pth"
MODEL_HASH = "abc123def456..."  # SHA256 checksum - replace with actual hash

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "style_shift" / "models"


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 8192
) -> None:
    """Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        dest: Destination file path
        chunk_size: Download chunk size in bytes
    """
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


def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    """Verify downloaded file integrity using SHA256.
    
    Args:
        file_path: Path to file
        expected_hash: Expected SHA256 hash
    
    Returns:
        True if hash matches, False otherwise
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    return actual_hash == expected_hash


def download_model(
    url: str = MODEL_URL,
    cache_dir: Optional[Path] = None,
    verify_hash: bool = True
) -> Path:
    """Download pre-trained decoder model.
    
    Args:
        url: Model URL
        cache_dir: Cache directory. If None, uses default
        verify_hash: Whether to verify checksum. Default: True
    
    Returns:
        Path to downloaded model file
    
    Raises:
        RuntimeError: If download fails or hash verification fails
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    model_path = cache_dir / "decoder.pth"
    
    # Check if already downloaded
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        
        if verify_hash:
            if verify_checksum(model_path, MODEL_HASH):
                print("✓ Checksum verified")
                return model_path
            else:
                print("⚠ Checksum mismatch, re-downloading...")
                model_path.unlink()
        else:
            return model_path
    
    # Download model
    print(f"Downloading model from {url}...")
    
    try:
        download_file(url, model_path)
        print(f"✓ Downloaded to {model_path}")
        
        # Verify checksum
        if verify_hash:
            print("Verifying checksum...")
            if not verify_checksum(model_path, MODEL_HASH):
                model_path.unlink()
                raise RuntimeError("Checksum verification failed!")
            print("✓ Checksum verified")
        
        return model_path
    
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Download failed: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained StyleShift models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Download with defaults
  %(prog)s --cache-dir /models  # Custom cache directory
  %(prog)s --url URL            # Custom URL
        """
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        help=f"Model URL (default: {MODEL_URL})"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification"
    )
    
    args = parser.parse_args()
    
    try:
        model_path = download_model(
            url=args.url or MODEL_URL,
            cache_dir=args.cache_dir,
            verify_hash=not args.no_verify
        )
        print(f"\n✓ Model ready: {model_path}")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
