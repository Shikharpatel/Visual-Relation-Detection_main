import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    print(f"Downloading {url}\nto {output_path}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc="Extracting"):
            zip_ref.extract(member=file, path=extract_to)
    print(f"Done extracting {zip_path}.")

def main():
    base_dir = Path(__file__).parent.parent
    img_dir = base_dir / 'data' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 10GB and 5GB
    files_to_download = [
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "VG_100K.zip", "VG_100K"),
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "VG_100K_2.zip", "VG_100K_2")
    ]
    
    print("WARNING: This script will download ~15GB of data. Ensure you have a stable connection and enough disk space.")
    
    for url, filename, folder_name in files_to_download:
        zip_path = img_dir / filename
        target_folder = img_dir / folder_name
        
        # Check if the extracted folder already exists
        if target_folder.exists() and len(os.listdir(target_folder)) > 0:
            print(f"Skipping {filename}: The folder {target_folder} already exists with contents.")
            continue
            
        # Download
        if not zip_path.exists():
            try:
                download_url(url, zip_path)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                continue
        else:
            print(f"Skipping download, {filename} already exists.")
            
        # Extract
        try:
            extract_zip(zip_path, img_dir)
            # Remove zip to save huge disk space
            print(f"Cleaning up {zip_path} to save 5-10GB of storage...")
            os.remove(zip_path)
        except Exception as e:
            print(f"Error extracting {zip_path}. It may be corrupted: {e}")

    print("\nAll VG Image downloads and extraction finished successfully!")

if __name__ == '__main__':
    main()
