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
    print(f"Downloading {url} to {output_path}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc="Extracting"):
            zip_ref.extract(member=file, path=extract_to)
    print(f"Done extracting {zip_path}.")

def main():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    urls = [
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
    ]
    
    for url in urls:
        filename = url.split('/')[-1]
        zip_path = raw_dir / filename
        
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
        json_filename = filename.replace('.zip', '')
        json_path = raw_dir / json_filename
        if not json_path.exists():
            extract_zip(zip_path, raw_dir)
            # Remove zip file to save space
            if zip_path.exists():
                os.remove(zip_path)
        else:
            print(f"Skipping extraction, {json_filename} already exists.")

if __name__ == '__main__':
    main()
