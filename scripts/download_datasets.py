#!/usr/bin/env python3
"""Download MS-COCO and WikiArt datasets for training StyleShift."""

import argparse
import os
import zipfile
from pathlib import Path


def download_file(url, dest, chunk_size=8192):
    """Download a file with progress."""
    import requests
    from tqdm import tqdm
    
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


def download_mscoco(data_dir='data/coco', year='2014'):
    """Download MS-COCO dataset."""
    print(f"Downloading MS-COCO {year}...")
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # MS-COCO 2014 train images
    urls = [
        f'http://images.cocodataset.org/zips/train{year}.zip',
    ]
    
    for url in urls:
        filename = url.split('/')[-1]
        dest = data_dir / filename
        
        if dest.exists():
            print(f"  Skipping {filename} (already exists)")
        else:
            print(f"  Downloading {filename}...")
            download_file(url, dest)
            
            # Extract
            print(f"  Extracting {filename}...")
            with zipfile.ZipFile(dest, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip
            dest.unlink()
    
    print(f"MS-COCO downloaded to {data_dir}")


def download_wikiart(data_dir='data/wikiart'):
    """Download WikiArt dataset."""
    print("Downloading WikiArt dataset...")
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # WikiArt can be downloaded from various sources
    # This is a placeholder - actual URL would need to be provided
    print("Note: WikiArt dataset download requires manual setup")
    print("Please download from:")
    print("  - https://www.kaggle.com/c/painting-classification/data")
    print("  - Or use: kaggle datasets download -d wanyi/painting-classification")
    print()
    
    # Create placeholder directory structure
    (data_dir / 'content').mkdir(exist_ok=True)
    
    print("WikiArt directory created at", data_dir)


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    
    parser.add_argument('--mscoco-dir', type=str, default='data/coco',
                        help='Directory for MS-COCO dataset')
    parser.add_argument('--wikiart-dir', type=str, default='data/wikiart',
                        help='Directory for WikiArt dataset')
    parser.add_argument('--mscoco-year', type=str, default='2014',
                        help='MS-COCO year (2014 or 2017)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("StyleShift Dataset Downloader")
    print("="*70)
    print()
    
    # Download MS-COCO
    download_mscoco(args.mscoco_dir, args.mscoco_year)
    print()
    
    # Download WikiArt
    download_wikiart(args.wikiart_dir)
    print()
    
    print("="*70)
    print("Dataset download complete!")
    print("="*70)
    print()
    print("Directory structure:")
    print(f"  {args.mscoco_dir}/")
    print(f"    └── train{args.mscoco_year}/")
    print(f"        ├── [80K images]")
    print(f"  {args.wikiart_dir}/")
    print(f"    └── content/")
    print(f"        ├── [80K images]")


if __name__ == "__main__":
    main()
