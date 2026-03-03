import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import open_clip
from sklearn.decomposition import PCA

from encoders import ImageEncoder, TextEncoder
from ebm import EBMHead
from mcmc import run_mcmc

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained weights
ckpt = torch.load("checkpoints/vibe_lean.pt", map_location=device)
ebm  = EBMHead().to(device)
ebm.load_state_dict(ckpt["ebm"])
ebm.eval()

img_enc  = ImageEncoder(device)
txt_enc  = TextEncoder(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Pick a sample image
img_path = list(Path("data/coco_val2017/images").glob("*.jpg"))[42]
image    = img_enc.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
img_emb  = img_enc(image)

# Encode reference captions as anchor points
captions = [
    "a person cooking in the kitchen",
    "a dog running in the park",
    "a car driving on the road",
    "people sitting at a table",
    "a cat sleeping on a couch",
]
tokens   = tokenizer(captions)
txt_embs = txt_enc(tokens).cpu().numpy()

# Run MCMC from random start with smaller step size
txt_emb_random = F.normalize(torch.randn(1, 512), dim=-1).to(device)
trajectory     = run_mcmc(img_emb, txt_emb_random, ebm,
                          n_steps=50, step_size=0.001, anneal=True)

# Extract trajectory embeddings + energies
traj_embs    = np.vstack([s["txt_emb"].numpy() for s in trajectory])
traj_energies = [s["energy"] for s in trajectory]

# PCA: fit on reference captions + trajectory
all_embs = np.vstack([txt_embs, traj_embs])
pca      = PCA(n_components=2)
all_2d   = pca.fit_transform(all_embs)

ref_2d  = all_2d[:len(captions)]
traj_2d = all_2d[len(captions):]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: PCA trajectory
ax = axes[0]
sc = ax.scatter(traj_2d[:, 0], traj_2d[:, 1],
                c=range(len(traj_2d)), cmap="plasma", s=30, zorder=3)
ax.plot(traj_2d[:, 0], traj_2d[:, 1], "gray", alpha=0.4, zorder=2)
ax.scatter(traj_2d[0, 0],  traj_2d[0, 1],  s=120, c="blue",  zorder=5, label="Start")
ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], s=120, c="green", zorder=5, label="End")
for i, cap in enumerate(captions):
    ax.scatter(ref_2d[i, 0], ref_2d[i, 1], s=80, c="red", marker="*", zorder=4)
    ax.annotate(cap[:20], ref_2d[i], fontsize=7, ha="left")
plt.colorbar(sc, ax=ax, label="MCMC Step")
ax.set_title("MCMC Trajectory in PCA Embedding Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

# Right: Energy over steps
ax = axes[1]
ax.plot(traj_energies, color="darkorange", linewidth=2)
ax.axhline(y=min(traj_energies), color="green", linestyle="--", alpha=0.5, label="Min energy")
ax.set_title("Energy over MCMC Steps")
ax.set_xlabel("Step")
ax.set_ylabel("Energy E(x, y)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("assets/mcmc_trajectory.png", dpi=150)
plt.show()
print("Saved to assets/mcmc_trajectory.png")