"""
Download the FULL Visual Genome images from multiple mirrors.
Tries official VG site first, then Hugging Face fallback.
"""
import os
import sys
import zipfile
from pathlib import Path

def download_with_urllib(url, output_path):
    """Download with progress bar using urllib."""
    import urllib.request
    from tqdm import tqdm

    class ProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)

    with ProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    """Extract zip with progress."""
    from tqdm import tqdm
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for m in tqdm(members, desc="Extracting"):
            zf.extract(m, extract_to)
    print(f"Done. Deleting ZIP to save space...")
    os.remove(zip_path)

def main():
    base_dir = Path(__file__).parent.parent
    img_dir = base_dir / 'data' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    # Check current state
    vg100k = img_dir / 'VG_100K'
    current_count = len(list(vg100k.glob('*.jpg'))) if vg100k.exists() else 0
    print(f"Current VG_100K image count: {current_count}")

    if current_count >= 90000:
        print("VG_100K appears complete. Nothing to do.")
        return

    # Mirror list (try each in order)
    mirrors = [
        {
            'name': 'Official VG (homes.cs.washington.edu)',
            'url': 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/images.zip',
        },
        {
            'name': 'Stanford CS',
            'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip',
        },
    ]

    zip_path = img_dir / 'VG_100K_full.zip'

    for mirror in mirrors:
        print(f"\nTrying {mirror['name']}...")
        print(f"URL: {mirror['url']}")
        try:
            download_with_urllib(mirror['url'], str(zip_path))
            
            # Verify the zip
            if zip_path.exists() and zip_path.stat().st_size > 1_000_000_000:  # > 1GB
                print(f"Download successful! Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
                
                # Remove old incomplete VG_100K
                if vg100k.exists():
                    import shutil
                    print(f"Removing old VG_100K ({current_count} files)...")
                    shutil.rmtree(vg100k)
                
                # Extract
                extract_zip(str(zip_path), str(img_dir))
                
                # Verify
                new_count = len(list(vg100k.glob('*.jpg'))) if vg100k.exists() else 0
                print(f"\nNew VG_100K image count: {new_count}")
                
                if new_count > current_count:
                    print("SUCCESS! Got more images.")
                    return
                else:
                    print("WARNING: Same or fewer images after re-download.")
                    continue
            else:
                print(f"Download too small ({zip_path.stat().st_size if zip_path.exists() else 0} bytes). Trying next mirror...")
                if zip_path.exists():
                    os.remove(zip_path)
                continue

        except Exception as e:
            print(f"Failed: {e}")
            if zip_path.exists():
                os.remove(zip_path)
            continue

    print("\nAll mirrors exhausted. Consider using Hugging Face:")
    print("  pip install huggingface_hub")
    print("  huggingface-cli download BoyangZ/VisualGenome_VG_100K_1_and_2 --local-dir data/images")

if __name__ == '__main__':
    main()
