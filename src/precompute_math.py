import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import open_clip

from encoders import ImageEncoder, TextEncoder
from model import VIBEModel
from mcmc import run_mcmc
from visualize_attention import get_attention_maps, attention_rollout, overlay_attention

device   = "cuda" if torch.cuda.is_available() else "cpu"
ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")

# ── Load model ─────────────────────────────────────────────
print("Loading model...")
ckpt  = torch.load("checkpoints/vibe_lean.pt", map_location=device)
model = VIBEModel(device=device, use_projection=True)
model.ebm.load_state_dict(ckpt["ebm"])
model.img_proj.load_state_dict(ckpt["img_proj"])
model.txt_proj.load_state_dict(ckpt["txt_proj"])
model.eval()

img_enc   = ImageEncoder(device)
txt_enc   = TextEncoder(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ── Fixed sample images ────────────────────────────────────
# To this:
import random
random.seed(42)
img_paths     = list(Path("data/coco_val2017/images").glob("*.jpg"))
SAMPLE_IMAGES = random.sample(img_paths, 6)

# ── Fixed candidate captions ───────────────────────────────
CAPTIONS = [
    "a red double decker bus on a street",
    "a dog playing in the park",
    "a person cooking in the kitchen",
    "a cat sitting on a couch",
    "a group of people at a sports event",
    "a bicycle parked near a building",
    "a bowl of fresh fruit on a table",
    "an airplane flying in the sky",
    "a child playing with toys",
    "a mountain landscape with snow",
]

# ══════════════════════════════════════════════════════════
# CACHE 1 — Energy scores + τ experiments
# ══════════════════════════════════════════════════════════
print("Computing energy cache...")
math_cache = {}

for img_idx, img_path in enumerate(SAMPLE_IMAGES):
    print(f"  Image {img_idx+1}/6: {img_path.name}")
    image      = Image.open(img_path).convert("RGB")
    img_tensor = img_enc.preprocess(image).unsqueeze(0)

    with torch.no_grad():
        img_emb, _ = model.encode(img_tensor, tokenizer([""]))

    # Encode all captions
    tokens = tokenizer(CAPTIONS)
    with torch.no_grad():
        _, txt_embs = model.encode(
            img_tensor.expand(len(CAPTIONS), -1, -1, -1), tokens
        )

    # Raw dot products (before τ scaling)
    dots = (img_emb @ txt_embs.T).squeeze().cpu().detach()

    # Energy scores at different τ values
    tau_values  = [0.01, 0.03, 0.055, 0.1, 0.3, 0.5, 1.0]
    tau_results = {}
    for tau in tau_values:
        energies = (-(dots / tau)).tolist()
        logits   = -torch.tensor(energies)
        probs    = F.softmax(logits, dim=0).tolist()
        tau_results[tau] = {
            "energies" : energies,
            "probs"    : probs,
            "entropy"  : float(-sum(p * np.log(p+1e-9) for p in probs)),
            "top1_conf": float(max(probs))
        }

    math_cache[img_idx] = {
        "img_path"   : str(img_path),
        "dots"       : dots.tolist(),
        "tau_results": tau_results,
        "trained_tau": float(model.ebm.tau.item()),
        "img_emb"    : img_emb.cpu(),
        "txt_embs"   : txt_embs.cpu()
    }

torch.save({
    "math_cache": math_cache,
    "captions"  : CAPTIONS
}, "assets/math_cache.pt")
print("  Saved assets/math_cache.pt")

# ══════════════════════════════════════════════════════════
# CACHE 2 — Per-layer attention maps
# ══════════════════════════════════════════════════════════
print("Computing attention cache...")
attention_cache = {}

def get_per_layer_attention(img_enc, image_tensor):
    """Extract attention map per transformer layer."""
    layer_attentions = []

    for layer_idx, block in enumerate(img_enc.encoder.transformer.resblocks):
        attentions = []

        def make_hook(mod):
            orig = mod.forward
            def patched(*args, **kwargs):
                kwargs["need_weights"]        = True
                kwargs["average_attn_weights"] = False
                out, w = orig(*args, **kwargs)
                if w is not None:
                    attentions.append(w.detach().cpu())
                return out, w
            return patched

        orig_fwd     = block.attn.forward
        block.attn.forward = make_hook(block.attn)

        with torch.no_grad():
            img_enc.encoder(image_tensor.to(device))

        block.attn.forward = orig_fwd

        if attentions:
            attn_avg  = attentions[0].mean(dim=1)[0]
            grid_size = int((attn_avg.shape[0] - 1) ** 0.5)
            cls_attn  = attn_avg[0, 1:].reshape(grid_size, grid_size).numpy()
            cls_attn  = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-9)
            layer_attentions.append(cls_attn)

    return layer_attentions

for img_idx, img_path in enumerate(SAMPLE_IMAGES):
    print(f"  Image {img_idx+1}/6: {img_path.name}")
    image       = Image.open(img_path).convert("RGB")
    img_tensor  = img_enc.preprocess(image).unsqueeze(0)
    layer_maps  = get_per_layer_attention(img_enc, img_tensor)
    full_attn   = attention_rollout(get_attention_maps(img_enc, img_tensor))

    attention_cache[img_idx] = {
        "img_path"   : str(img_path),
        "layer_maps" : layer_maps,    # list of 12 (7x7) arrays
        "full_rollout": full_attn     # (7x7) array
    }

torch.save(attention_cache, "assets/attention_cache.pt")
print("  Saved assets/attention_cache.pt")

# ══════════════════════════════════════════════════════════
# CACHE 3 — Real MCMC trajectories
# ══════════════════════════════════════════════════════════
print("Computing MCMC cache...")
mcmc_cache = {}

STEP_SIZES  = [0.0001, 0.001, 0.01]
data        = torch.load("checkpoints/embeddings.pt", map_location="cpu")
all_txt_embs = data["txt_embs"]

for img_idx in range(len(SAMPLE_IMAGES)):
    print(f"  Image {img_idx+1}/6")
    img_emb = math_cache[img_idx]["img_emb"].to(device)
    runs    = {}

    for alpha in STEP_SIZES:
        txt_init   = F.normalize(torch.randn(1, 512), dim=-1).to(device)
        trajectory = run_mcmc(img_emb, txt_init, model.ebm,
                              n_steps=50, step_size=alpha, anneal=True)

        # Find nearest caption at each step
        nearest_captions = []
        for step in trajectory:
            emb  = step["txt_emb"]
            sims = (emb @ all_txt_embs.T).squeeze()
            best = sims.argmax().item()
            nearest_captions.append(best)

        runs[alpha] = {
            "energies"        : [s["energy"] for s in trajectory],
            "nearest_captions": nearest_captions
        }

    mcmc_cache[img_idx] = runs

torch.save({
    "mcmc_cache"  : mcmc_cache,
    "step_sizes"  : STEP_SIZES
}, "assets/mcmc_cache.pt")
print("  Saved assets/mcmc_cache.pt")

print("\nAll caches computed successfully!")
print("  assets/math_cache.pt")
print("  assets/attention_cache.pt")
print("  assets/mcmc_cache.pt")