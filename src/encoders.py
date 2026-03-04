import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ── Shared CLIP model (loaded once) ───────────────────────
_clip_model   = None
_preprocess   = None
_tokenizer    = None

def get_clip(device="cpu"):
    global _clip_model, _preprocess, _tokenizer
    if _clip_model is None:
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        for param in _clip_model.parameters():
            param.requires_grad = False
        _clip_model.to(device)
        _clip_model.eval()
    return _clip_model, _preprocess, _tokenizer


class ImageEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        model, preprocess, _ = get_clip(device)
        self.encoder    = model.visual
        self.preprocess = preprocess

    def forward(self, images):
        with torch.no_grad():
            feats = self.encoder(images.to(self.device))
        return F.normalize(feats, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        model, _, tokenizer = get_clip(device)
        self.encoder   = model
        self.tokenizer = tokenizer

    def forward(self, tokens):
        with torch.no_grad():
            feats = self.encoder.encode_text(tokens.to(self.device))
        return F.normalize(feats, dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)