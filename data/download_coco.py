import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent / "coco_val2017"
IMG_DIR  = DATA_DIR / "images"
ANN_DIR  = DATA_DIR / "annotations"

# URLs
URLS = {
    "images"      : "http://images.cocodataset.org/zips/val2017.zip",
    "annotations" : "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download_file(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded")
        return
    print(f"  Downloading {dest.name} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    zip_path.unlink()
    print(f"  Done → {extract_to}")

if __name__ == "__main__":
    # Download
    img_zip = DATA_DIR / "val2017.zip"
    ann_zip = DATA_DIR / "annotations.zip"
    download_file(URLS["images"],      img_zip)
    download_file(URLS["annotations"], ann_zip)

    # Extract
    extract_zip(img_zip, DATA_DIR)
    extract_zip(ann_zip, DATA_DIR)

    # Verify
    imgs = list(IMG_DIR.glob("*.jpg"))
    print(f"\n  Images   : {len(imgs)}")
    print(f"  Captions : {(ANN_DIR / 'captions_val2017.json').exists()}")
    print("\nDownload complete!")