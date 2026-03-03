import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from umap import UMAP

# Load cached embeddings
data     = torch.load("checkpoints/embeddings.pt", map_location="cpu")
img_embs = data["img_embs"].numpy()
txt_embs = data["txt_embs"].numpy()

# Use 500 samples for speed
N          = 500
img_sample = img_embs[:N]
txt_sample = txt_embs[:N]

print("Running UMAP (this may take ~1 min)...")

# Fit UMAP on combined image + text embeddings
all_embs = np.vstack([img_sample, txt_sample])
labels   = np.array(["image"] * N + ["text"] * N)
reducer  = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                metric="cosine", random_state=42)
embs_2d  = reducer.fit_transform(all_embs)

img_2d = embs_2d[:N]
txt_2d = embs_2d[N:]

# Compute per-pair cosine similarity for coloring
sims = (img_sample * txt_sample).sum(axis=1)
sims = (sims - sims.min()) / (sims.max() - sims.min())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Image vs Text embeddings
ax = axes[0]
ax.scatter(img_2d[:, 0], img_2d[:, 1], c="royalblue", s=15,
           alpha=0.6, label="Image embeddings")
ax.scatter(txt_2d[:, 0], txt_2d[:, 1], c="tomato",    s=15,
           alpha=0.6, label="Text embeddings")

# Draw lines connecting matched pairs
for i in range(0, N, 25):
    ax.plot([img_2d[i, 0], txt_2d[i, 0]],
            [img_2d[i, 1], txt_2d[i, 1]],
            "gray", alpha=0.3, linewidth=0.8)

ax.set_title("UMAP: Image vs Text Embedding Space\n(lines = matched pairs)")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend()
ax.grid(alpha=0.2)

# Right: Colored by alignment score
ax = axes[1]
sc = ax.scatter(img_2d[:, 0], img_2d[:, 1], c=sims,
                cmap="RdYlGn", s=20, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Image-Text Cosine Similarity")
ax.set_title("UMAP: Image Embeddings\nColored by alignment with matched text")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.grid(alpha=0.2)

plt.suptitle("Embedding Space Geometry — VIBE Model", fontsize=13)
plt.tight_layout()
plt.savefig("assets/umap_embeddings.png", dpi=150)
plt.show()
print("Saved to assets/umap_embeddings.png")