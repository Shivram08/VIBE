import sys
sys.path.insert(0, "src")

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import VIBEModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model + embeddings
ckpt     = torch.load("checkpoints/vibe_lean.pt",   map_location=device)
data     = torch.load("checkpoints/embeddings.pt",  map_location="cpu")
model    = VIBEModel(device=device, use_projection=True)
model.ebm.load_state_dict(ckpt["ebm"])
model.eval()

img_embs = data["img_embs"].to(device)
txt_embs = data["txt_embs"].to(device)

# Compute energy matrix on a subset (500x500 for memory)
N = 500
with torch.no_grad():
    E_matrix = model.ebm(img_embs[:N], txt_embs[:N]).cpu()

E_matched   = E_matrix.diag().numpy()
mask        = ~torch.eye(N, dtype=bool)
E_unmatched = E_matrix[mask].numpy()

# Sample unmatched for visualization (too many points otherwise)
idx         = np.random.choice(len(E_unmatched), size=5000, replace=False)
E_unmatched = E_unmatched[idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram of energy distributions
ax = axes[0]
ax.hist(E_unmatched, bins=80, alpha=0.6, color="red",   label="Unmatched pairs", density=True)
ax.hist(E_matched,   bins=30, alpha=0.8, color="green", label="Matched pairs",   density=True)
ax.axvline(E_matched.mean(),   color="darkgreen", linestyle="--",
           label=f"Mean matched: {E_matched.mean():.2f}")
ax.axvline(E_unmatched.mean(), color="darkred",   linestyle="--",
           label=f"Mean unmatched: {E_unmatched.mean():.2f}")
ax.set_xlabel("Energy E(x, y)")
ax.set_ylabel("Density")
ax.set_title("Energy Distribution: Matched vs Unmatched Pairs")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Right: Energy matrix heatmap (50x50 subset)
ax = axes[1]
im = ax.imshow(E_matrix[:50, :50].numpy(), cmap="RdYlGn_r", aspect="auto")
ax.set_title("Energy Matrix (50×50 subset)\nDiagonal = matched pairs (green=low energy)")
ax.set_xlabel("Text index")
ax.set_ylabel("Image index")
plt.colorbar(im, ax=ax, label="Energy")

plt.tight_layout()
plt.savefig("assets/energy_analysis.png", dpi=150)
plt.show()
print("Saved to assets/energy_analysis.png")