import torch
import torch.nn as nn
from encoders import ImageEncoder, TextEncoder, ProjectionHead
from ebm import EBMHead
from losses import InfoNCELoss


class VIBEModel(nn.Module):
    """
    Full VIBE model:
      1. Encode image + text with frozen CLIP encoders
      2. Optionally project through learnable MLP heads
      3. Compute energy E(x, y) via EBM head
      4. Train with InfoNCE loss
    """
    def __init__(self, device="cuda", use_projection=True):
        super().__init__()
        self.device         = device
        self.use_projection = use_projection

        self.img_encoder = ImageEncoder(device)
        self.txt_encoder = TextEncoder(device)
        self.ebm         = EBMHead().to(device)
        self.loss_fn     = InfoNCELoss()

        if use_projection:
            self.img_proj = ProjectionHead().to(device)
            self.txt_proj = ProjectionHead().to(device)

    def encode(self, images, tokens):
        img_emb = self.img_encoder(images)
        txt_emb = self.txt_encoder(tokens)

        if self.use_projection:
            img_emb = self.img_proj(img_emb)
            txt_emb = self.txt_proj(txt_emb)

        return img_emb, txt_emb

    def forward(self, images, tokens):
        img_emb, txt_emb = self.encode(images, tokens)
        energy           = self.ebm(img_emb, txt_emb)
        loss             = self.loss_fn(energy)
        return loss, energy

    def get_learnable_params(self):
        params = [{"params": self.ebm.parameters(), "lr": 1e-3}]
        if self.use_projection:
            params += [
                {"params": self.img_proj.parameters(), "lr": 1e-4},
                {"params": self.txt_proj.parameters(), "lr": 1e-4}
            ]
        return params


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import DataLoader
    import sys
    sys.path.insert(0, "src")
    from dataset import COCODataset
    import open_clip

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")

    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer        = open_clip.get_tokenizer("ViT-B-32")

    dataset    = COCODataset(ann_path, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = VIBEModel(device=device, use_projection=True)

    imgs, txts, _ = next(iter(dataloader))
    loss, energy  = model(imgs, txts)

    print(f"Loss                  : {loss.item():.4f}")
    print(f"Expected ≈ log(4)     : {__import__('math').log(4):.4f}")
    print(f"Energy matrix shape   : {energy.shape}")
    print(f"Temperature τ         : {model.ebm.tau.item():.4f}")
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters  : {learnable:,}")
    print("VIBEModel OK!")