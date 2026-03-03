import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from dataset import load_coco_index

device   = "cuda" if torch.cuda.is_available() else "cpu"
ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")

# Load embeddings + index
data     = torch.load("checkpoints/embeddings.pt", map_location="cpu")
img_embs = data["img_embs"]
txt_embs = data["txt_embs"]
index    = load_coco_index(ann_path)
img_ids  = list(index.keys())

# Similarity matrix
sim = img_embs @ txt_embs.T
N   = sim.shape[0]

# Find rank of correct caption for each image
ranks = []
for i in range(N):
    sorted_idx = sim[i].argsort(descending=True)
    rank       = (sorted_idx == i).nonzero(as_tuple=True)[0].item()
    ranks.append(rank)
ranks = np.array(ranks)

# Top 8 worst retrievals (highest rank = most wrong)
worst_idx = np.argsort(ranks)[::-1][:8]

fig, axes = plt.subplots(2, 8, figsize=(22, 6))

for col, idx in enumerate(worst_idx):
    img_id  = img_ids[idx]
    entry   = index[img_id]
    image   = Image.open(entry["image_path"]).convert("RGB")

    # Correct caption
    correct_cap = entry["captions"][0]

    # What the model retrieved instead (top-1 wrong)
    top1_idx    = sim[idx].argsort(descending=True)[0].item()
    if top1_idx == idx:
        top1_idx = sim[idx].argsort(descending=True)[1].item()
    retrieved_cap = index[img_ids[top1_idx]]["captions"][0]

    # Top row: image
    axes[0, col].imshow(image.resize((150, 150)))
    axes[0, col].set_title(f"Rank: {ranks[idx]+1}", fontsize=8, color="red")
    axes[0, col].axis("off")

    # Bottom row: correct vs retrieved caption
    caption_text = f"✓ {correct_cap[:50]}...\n\n✗ {retrieved_cap[:50]}..."
    axes[1, col].text(0.5, 0.5, caption_text, ha="center", va="center",
                      fontsize=6, wrap=True,
                      transform=axes[1, col].transAxes,
                      bbox=dict(boxstyle="round", facecolor="lightyellow"))
    axes[1, col].axis("off")

plt.suptitle("Top-8 Failure Cases — Correct caption vs Model's top-1 retrieval",
             fontsize=11)
plt.tight_layout()
plt.savefig("assets/failure_cases.png", dpi=150)
plt.show()
print("Saved to assets/failure_cases.png")

# Print rank distribution
print(f"\nRank distribution:")
print(f"  Rank 1 (R@1)  : {(ranks == 0).sum()} / {N}")
print(f"  Rank ≤ 5      : {(ranks < 5).sum()} / {N}")
print(f"  Rank ≤ 10     : {(ranks < 10).sum()} / {N}")
print(f"  Worst rank    : {ranks.max() + 1}")
print(f"  Median rank   : {np.median(ranks) + 1:.1f}")