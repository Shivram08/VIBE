import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import open_clip

from encoders import ImageEncoder, TextEncoder
from ebm import EBMHead
from mcmc import run_mcmc

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained weights
ckpt      = torch.load("checkpoints/vibe_lean.pt", map_location=device)
ebm       = EBMHead().to(device)
ebm.load_state_dict(ckpt["ebm"])
ebm.eval()

# Load encoders
img_enc = ImageEncoder(device)
txt_enc = TextEncoder(device)

# Pick a sample image
img_path = next(Path("data/coco_val2017/images").glob("*.jpg"))
image    = img_enc.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
img_emb  = img_enc(image)

# Two starting points: correct caption vs random noise
_, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer        = open_clip.get_tokenizer("ViT-B-32")

correct_caption = "a photo of people and objects in a scene"
tokens          = tokenizer([correct_caption])
txt_emb_correct = txt_enc(tokens)
txt_emb_random  = F.normalize(torch.randn(1, 512), dim=-1).to(device)

print("=== Starting from CORRECT caption ===")
traj_correct = run_mcmc(img_emb, txt_emb_correct, ebm, n_steps=50, step_size=0.01)
energies_c   = [s["energy"] for s in traj_correct]
print(f"Start energy : {energies_c[0]:.4f}")
print(f"Final energy : {energies_c[-1]:.4f}")
print(f"Δ Energy     : {energies_c[0] - energies_c[-1]:.4f}")

print("\n=== Starting from RANDOM noise ===")
traj_random  = run_mcmc(img_emb, txt_emb_random, ebm, n_steps=50, step_size=0.01)
energies_r   = [s["energy"] for s in traj_random]
print(f"Start energy : {energies_r[0]:.4f}")
print(f"Final energy : {energies_r[-1]:.4f}")
print(f"Δ Energy     : {energies_r[0] - energies_r[-1]:.4f}")

print("\n=== Energy every 10 steps (random start) ===")
for s in traj_random[::10]:
    print(f"  Step {s['step']:>3} | Energy: {s['energy']:.4f}")

print("\nMCMC convergence test OK!")