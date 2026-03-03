import torch
from PIL import Image
from pathlib import Path
from encoders import ImageEncoder, TextEncoder, ProjectionHead

device = "cuda" if torch.cuda.is_available() else "cpu"

# Init
img_enc = ImageEncoder(device)
txt_enc = TextEncoder(device)
img_proj = ProjectionHead().to(device)
txt_proj = ProjectionHead().to(device)

# Load 4 images and 4 captions
img_dir = Path("data/coco_val2017/images")
paths   = list(img_dir.glob("*.jpg"))[:4]
images  = torch.stack([
    img_enc.preprocess(Image.open(p).convert("RGB")) for p in paths
]).to(device)

captions = [
    "a dog playing in the park",
    "a cat sitting on a couch",
    "a person riding a bicycle",
    "a bowl of fresh fruit"
]
tokens = txt_enc.tokenizer(captions)

# Encode
img_embs = img_enc(images)
txt_embs = txt_enc(tokens)
img_proj_embs = img_proj(img_embs)
txt_proj_embs = txt_proj(txt_embs)

# Similarity matrix (4x4)
sim_matrix = img_embs @ txt_embs.T
print("Cosine Similarity Matrix (images x captions):")
print(sim_matrix.round(decimals=3))
print()
print(f"Image embeddings shape    : {img_embs.shape}")
print(f"Text embeddings shape     : {txt_embs.shape}")
print(f"All L2 norms == 1.0       : {torch.allclose(img_embs.norm(dim=-1), torch.ones(4).to(device))}")
print(f"Projected shapes match    : {img_proj_embs.shape == txt_proj_embs.shape}")
print("Full encoder pipeline OK!")