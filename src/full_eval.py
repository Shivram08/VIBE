import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import open_clip

from encoders import ImageEncoder, TextEncoder, ProjectionHead
from ebm import EBMHead
from model import VIBEModel
from dataset import COCODataset

device   = "cuda" if torch.cuda.is_available() else "cpu"
ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")

# Load model
ckpt      = torch.load("checkpoints/vibe_lean.pt", map_location=device)
model     = VIBEModel(device=device, use_projection=True)
model.ebm.load_state_dict(ckpt["ebm"])
model.img_proj.load_state_dict(ckpt["img_proj"])
model.txt_proj.load_state_dict(ckpt["txt_proj"])
model.eval()

# Load full dataset
_, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer        = open_clip.get_tokenizer("ViT-B-32")
dataset          = COCODataset(ann_path, preprocess, tokenizer)
dataloader       = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

# Extract all embeddings
all_img_embs = []
all_txt_embs = []

print("Extracting embeddings...")
with torch.no_grad():
    for imgs, txts, _ in dataloader:
        img_emb, txt_emb = model.encode(imgs, txts)
        all_img_embs.append(img_emb.cpu())
        all_txt_embs.append(txt_emb.cpu())

img_embs = torch.cat(all_img_embs)  # (5000, 512)
txt_embs = torch.cat(all_txt_embs)  # (5000, 512)

print(f"Total embeddings: {img_embs.shape[0]}")

# Similarity matrix (5000 x 5000)
print("Computing similarity matrix...")
sim = img_embs @ txt_embs.T
N   = sim.shape[0]

# Recall@K
print("\n=== Image-to-Text Recall@K (full 5k) ===")
for k in [1, 5, 10]:
    topk    = sim.topk(k, dim=1).indices
    correct = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
    recall  = correct.float().mean().item()
    print(f"  R@{k:<3}: {recall:.4f}  ({int(recall*N)}/{N} correct)")

# Energy gap analysis
print("\n=== Energy Gap Analysis ===")
ebm         = model.ebm
img_embs_g  = img_embs.to(device)
txt_embs_g  = txt_embs.to(device)

with torch.no_grad():
    E_matrix  = ebm(img_embs_g, txt_embs_g).cpu()

E_matched   = E_matrix.diag()
mask        = ~torch.eye(N, dtype=bool)
E_unmatched = E_matrix[mask]

print(f"  Mean E(matched)   : {E_matched.mean():.4f}")
print(f"  Mean E(unmatched) : {E_unmatched.mean():.4f}")
print(f"  Energy gap        : {E_unmatched.mean() - E_matched.mean():.4f}")

# Save embeddings for later use
torch.save({
    "img_embs": img_embs,
    "txt_embs": txt_embs
}, "checkpoints/embeddings.pt")
print("\nEmbeddings saved to checkpoints/embeddings.pt")