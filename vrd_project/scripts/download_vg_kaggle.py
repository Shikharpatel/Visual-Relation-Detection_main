"""
Download Visual Genome Dataset from Kaggle
===========================================

Downloads the Visual Genome dataset from Kaggle and prepares it for preprocessing.

Dataset: https://www.kaggle.com/datasets/mathurinache/visual-genome

Usage:
    python scripts/download_vg_kaggle.py

Requirements:
    pip install kagglehub
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not installed. Run: pip install kagglehub")
    sys.exit(1)


def find_json_files(directory: Path) -> dict:
    """Find all JSON and ZIP files in directory tree."""
    files = {}
    for path in directory.rglob("*"):
        if path.suffix in ['.json', '.zip']:
            files[path.name.lower()] = path
    return files


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract a zip file and return path to extracted JSON."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
        # Find the extracted JSON file
        for name in zf.namelist():
            if name.endswith('.json'):
                return extract_to / name
    return None


def generate_image_data(objects_data: list) -> list:
    """
    Generate image_data.json from objects.json.

    Each image's dimensions are estimated from object bounding boxes.
    """
    print("Generating image_data.json from object annotations...")

    image_data = []

    for img_objects in objects_data:
        image_id = img_objects.get('image_id')
        objects = img_objects.get('objects', [])

        if not objects:
            continue

        # Estimate image dimensions from object bounding boxes
        max_x = 0
        max_y = 0

        for obj in objects:
            x = obj.get('x', 0)
            y = obj.get('y', 0)
            w = obj.get('w', obj.get('width', 0))
            h = obj.get('h', obj.get('height', 0))

            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # Add some padding and round to reasonable dimensions
        width = max(max_x + 50, 100)
        height = max(max_y + 50, 100)

        # Round to nearest 10
        width = int(round(width / 10) * 10)
        height = int(round(height / 10) * 10)

        image_data.append({
            'image_id': image_id,
            'width': width,
            'height': height,
            'url': f"https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg"
        })

    print(f"Generated metadata for {len(image_data)} images")
    return image_data


def main():
    # Output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Visual Genome Dataset Download")
    print("=" * 60)

    # Download from Kaggle
    print("\nDownloading from Kaggle (this may take a few minutes)...")
    print("Note: You may need to authenticate with Kaggle API key.")
    print("      Visit https://www.kaggle.com/settings to get your API key.")
    print("")

    try:
        download_path = kagglehub.dataset_download("mathurinache/visual-genome")
        download_path = Path(download_path)
        print(f"\nDownloaded to: {download_path}")
    except Exception as e:
        print(f"\nError downloading from Kaggle: {e}")
        print("\nTo authenticate with Kaggle:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place it in ~/.kaggle/kaggle.json (Linux/Mac)")
        print("   or C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")
        sys.exit(1)

    # Find downloaded files
    print("\nSearching for Visual Genome files...")
    files = find_json_files(download_path)

    print(f"Found files: {list(files.keys())}")

    # Look for objects.json or objects.json.zip
    objects_path = None
    relationships_path = None
    image_data_path = None

    # Check for direct JSON files first
    if 'objects.json' in files:
        objects_path = files['objects.json']
    elif 'objects.json.zip' in files:
        objects_path = extract_zip(files['objects.json.zip'], download_path)

    if 'relationships.json' in files:
        relationships_path = files['relationships.json']
    elif 'relationships.json.zip' in files:
        relationships_path = extract_zip(files['relationships.json.zip'], download_path)

    if 'image_data.json' in files:
        image_data_path = files['image_data.json']
    elif 'image_data.json.zip' in files:
        image_data_path = extract_zip(files['image_data.json.zip'], download_path)

    # Verify we have required files
    if not objects_path or not objects_path.exists():
        print("\nError: objects.json not found in download")
        print(f"Available files: {list(files.keys())}")
        sys.exit(1)

    if not relationships_path or not relationships_path.exists():
        print("\nError: relationships.json not found in download")
        print(f"Available files: {list(files.keys())}")
        sys.exit(1)

    # Copy/move files to data/raw
    print(f"\nCopying files to {output_dir}...")

    # Objects
    print(f"Loading {objects_path}...")
    with open(objects_path, 'r', encoding='utf-8') as f:
        objects_data = json.load(f)
    print(f"  Loaded {len(objects_data)} image entries")

    # Save to output
    with open(output_dir / 'objects.json', 'w', encoding='utf-8') as f:
        json.dump(objects_data, f)
    print(f"  Saved to {output_dir / 'objects.json'}")

    # Relationships
    print(f"Loading {relationships_path}...")
    with open(relationships_path, 'r', encoding='utf-8') as f:
        relationships_data = json.load(f)
    print(f"  Loaded {len(relationships_data)} image entries")

    with open(output_dir / 'relationships.json', 'w', encoding='utf-8') as f:
        json.dump(relationships_data, f)
    print(f"  Saved to {output_dir / 'relationships.json'}")

    # Image data - generate if not available
    if image_data_path and image_data_path.exists():
        print(f"Copying {image_data_path}...")
        shutil.copy(image_data_path, output_dir / 'image_data.json')
    else:
        print("\nimage_data.json not found, generating from objects...")
        image_data = generate_image_data(objects_data)

        with open(output_dir / 'image_data.json', 'w', encoding='utf-8') as f:
            json.dump(image_data, f)
        print(f"  Saved to {output_dir / 'image_data.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nFiles saved to {output_dir}/:")
    for f in output_dir.glob("*.json"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    print("\nNext step: Run preprocessing")
    print("  python -m src.data.vg_preprocess --config configs/default.yaml")


if __name__ == "__main__":
    main()
