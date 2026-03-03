# ╔══════════════════════════════════════════════════════╗
# ║         VIBE — Kaggle Training Script                ║
# ║  Run this in a Kaggle notebook with GPU enabled      ║
# ╚══════════════════════════════════════════════════════╝

# ── Cell 1: Install dependencies ──────────────────────
# !pip install open_clip_torch wandb -q

# ── Cell 2: Clone repo ────────────────────────────────
# !git clone https://github.com/Shivram08/VIBE.git
# %cd VIBE

# ── Cell 3: Download COCO ─────────────────────────────
# !python data/download_coco.py

# ── Cell 4: Login to W&B ──────────────────────────────
# import wandb
# wandb.login(key="YOUR_WANDB_API_KEY")

# ── Cell 5: Update config for Kaggle ──────────────────
import sys
sys.path.insert(0, "src")

from config import Config

cfg              = Config()
cfg.batch_size   = 256          # Kaggle P100/T4 has 16GB — use it
cfg.epochs       = 10
cfg.num_workers  = 4
cfg.run_name     = "vibe-kaggle-run-01"
print(cfg)

# ── Cell 6: Run training ──────────────────────────────
from train import train
train()

# ── Cell 7: Save outputs ──────────────────────────────
# After training, download checkpoints/best_model.pt
# from the Kaggle output panel on the right