import torch
import torch.nn as nn
import open_clip


class ImageEncoder(nn.Module):
    """
    Frozen CLIP ViT-B/32 image encoder.
    Returns L2-normalized embeddings of shape (B, 512).
    """
    def __init__(self, device="cuda"):
        super().__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.encoder = model.visual
        self.device  = device

        # Freeze all weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.to(device)
        self.encoder.eval()

    def forward(self, images):
        with torch.no_grad():
            feats = self.encoder(images.to(self.device))
        return nn.functional.normalize(feats, dim=-1)


class TextEncoder(nn.Module):
    """
    Frozen CLIP text transformer.
    Returns L2-normalized embeddings of shape (B, 512).
    """
    def __init__(self, device="cuda"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.encoder   = model
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device    = device

        # Freeze all weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.to(device)
        self.encoder.eval()

    def forward(self, tokens):
        with torch.no_grad():
            feats = self.encoder.encode_text(tokens.to(self.device))
        return nn.functional.normalize(feats, dim=-1)



class ProjectionHead(nn.Module):
    """
    Optional 2-layer MLP projection head (512 → 512).
    Adds learnable capacity on top of frozen CLIP embeddings.
    Toggle with use_projection flag in VIBEModel.
    """
    def __init__(self, in_dim=512, out_dim=512, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


if __name__ == "__main__":
    from PIL import Image
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    img_enc  = ImageEncoder(device)
    txt_enc  = TextEncoder(device)
    proj     = ProjectionHead().to(device)

    sample_path = next(Path("data/coco_val2017/images").glob("*.jpg"))
    image       = img_enc.preprocess(Image.open(sample_path).convert("RGB")).unsqueeze(0)
    tokens      = txt_enc.tokenizer(["a photo of a dog"])

    img_emb  = img_enc(image)
    txt_emb  = txt_enc(tokens)
    img_proj = proj(img_emb)
    txt_proj = proj(txt_emb)

    print(f"Image embedding shape      : {img_emb.shape}")
    print(f"Text embedding shape       : {txt_emb.shape}")
    print(f"Projected image shape      : {img_proj.shape}")
    print(f"Projected image L2 norm    : {img_proj.norm(dim=-1).item():.4f}")
    print(f"Projected text L2 norm     : {txt_proj.norm(dim=-1).item():.4f}")
    print("Projection heads OK!")