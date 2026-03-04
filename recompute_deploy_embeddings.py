import sys, torch, shutil
sys.path.insert(0, "src")
import open_clip
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import COCODataset
from model import VIBEModel
from huggingface_hub import HfApi

device   = "cuda"
ann_path = Path("deploy_check/annotations/captions_val2017.json")

# Also need the 100 images locally — copy from main data folder
deploy_img_dir = Path("deploy_check/images")
deploy_img_dir.mkdir(exist_ok=True)

import json
with open(ann_path) as f:
    raw = json.load(f)

img_names = {img["file_name"] for img in raw["images"]}
src_dir   = Path("data/coco_val2017/images")
copied    = 0
for name in img_names:
    src = src_dir / name
    if src.exists():
        shutil.copy(src, deploy_img_dir / name)
        copied += 1
print(f"Copied {copied}/100 images")

# Build dataset from deployed annotation + images
_, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Patch ann_path to point to deploy images
import dataset as ds_module
original_load = ds_module.load_coco_index

def patched_load(ann_path):
    index = original_load(ann_path)
    # Remap image paths to deploy folder
    for img_id in index:
        fname = index[img_id]["file_name"]
        index[img_id]["image_path"] = (deploy_img_dir / fname).resolve()
    return index

ds_module.load_coco_index = patched_load

dataset    = COCODataset(ann_path, preprocess, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
print(f"Dataset size: {len(dataset)}")

# Load model
ckpt  = torch.load("checkpoints/vibe_lean.pt", map_location=device)
model = VIBEModel(device=device, use_projection=True)
model.ebm.load_state_dict(ckpt["ebm"])
model.img_proj.load_state_dict(ckpt["img_proj"])
model.txt_proj.load_state_dict(ckpt["txt_proj"])
model.eval()

# Compute embeddings
all_img_embs, all_txt_embs = [], []
with torch.no_grad():
    for imgs, txts, _ in dataloader:
        img_emb, txt_emb = model.encode(imgs, txts)
        all_img_embs.append(img_emb.cpu())
        all_txt_embs.append(txt_emb.cpu())

img_embs = torch.cat(all_img_embs)
txt_embs = torch.cat(all_txt_embs)
print(f"Embeddings shape: {img_embs.shape}")

torch.save({
    "img_embs": img_embs,
    "txt_embs": txt_embs
}, "checkpoints/embeddings_deploy.pt")
print("Saved!")

# Upload to HuggingFace
api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/embeddings_deploy.pt",
    path_in_repo="embeddings_deploy.pt",
    repo_id="shiv0805/VIBE",
    repo_type="dataset"
)
print("Uploaded to HuggingFace!")