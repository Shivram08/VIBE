"""
Prepares a 100-image subset of COCO for Streamlit Cloud deployment.
Uploads images + annotations to HuggingFace dataset repo.
"""
import sys
sys.path.insert(0, "src")

import json
import shutil
import random
from pathlib import Path
from huggingface_hub import HfApi

# ── Config ─────────────────────────────────────────────────
HF_REPO    = "shiv0805/VIBE"   # replace with your username
N_IMAGES   = 100
SEED       = 42
DEPLOY_DIR = Path("deploy_data")

# ── Sample 100 images ──────────────────────────────────────
random.seed(SEED)
img_dir   = Path("data/coco_val2017/images")
all_imgs  = sorted(img_dir.glob("*.jpg"))
sampled   = random.sample(all_imgs, N_IMAGES)

# ── Copy images to deploy folder ───────────────────────────
deploy_img_dir = DEPLOY_DIR / "images"
deploy_img_dir.mkdir(parents=True, exist_ok=True)

print(f"Copying {N_IMAGES} images...")
for img_path in sampled:
    shutil.copy(img_path, deploy_img_dir / img_path.name)

# ── Filter annotations ─────────────────────────────────────
ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")
with open(ann_path) as f:
    raw = json.load(f)

sampled_names = {p.name for p in sampled}
sampled_ids   = {img["id"] for img in raw["images"]
                 if img["file_name"] in sampled_names}

filtered_images = [img for img in raw["images"]
                   if img["id"] in sampled_ids]
filtered_anns   = [ann for ann in raw["annotations"]
                   if ann["image_id"] in sampled_ids]

deploy_ann_dir = DEPLOY_DIR / "annotations"
deploy_ann_dir.mkdir(parents=True, exist_ok=True)

with open(deploy_ann_dir / "captions_val2017.json", "w") as f:
    json.dump({
        "images"     : filtered_images,
        "annotations": filtered_anns
    }, f)

print(f"Filtered annotations: {len(filtered_images)} images, {len(filtered_anns)} captions")

# ── Upload to HuggingFace ──────────────────────────────────
print("Uploading to HuggingFace...")
api = HfApi()

# Upload annotations
api.upload_file(
    path_or_fileobj=str(deploy_ann_dir / "captions_val2017.json"),
    path_in_repo="annotations/captions_val2017.json",
    repo_id=HF_REPO,
    repo_type="dataset"
)
print("Annotations uploaded!")

# Upload images one by one
imgs = list(deploy_img_dir.glob("*.jpg"))
for i, img_path in enumerate(imgs):
    api.upload_file(
        path_or_fileobj=str(img_path),
        path_in_repo=f"images/{img_path.name}",
        repo_id=HF_REPO,
        repo_type="dataset"
    )
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_IMAGES} images uploaded")

print(f"\nAll {N_IMAGES} images uploaded!")
print(f"Cleaning up deploy_data folder...")
shutil.rmtree(DEPLOY_DIR)
print("Done!")