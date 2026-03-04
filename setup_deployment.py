"""
Run once on Streamlit Cloud to download weights + data from HuggingFace.
Called automatically from app/main.py on first load.
"""
import os
import sys
from pathlib import Path

HF_REPO    = "shiv0805/vibe-weights"
CACHE_FLAG = Path(".deployment_ready")


def setup():
    if CACHE_FLAG.exists():
        return

    print("Setting up deployment environment...")

    from huggingface_hub import hf_hub_download, list_repo_files

    # ── Checkpoints ────────────────────────────────────
    Path("checkpoints").mkdir(exist_ok=True)

    print("Downloading model weights...")
    for fname in ["vibe_lean.pt"]:
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset",
            filename=fname,
            local_dir="checkpoints"
        )
        print(f"  {fname} downloaded!")

    # ── COCO subset ────────────────────────────────────
    img_dir = Path("data/coco_val2017/images")
    ann_dir = Path("data/coco_val2017/annotations")
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading COCO annotations...")
    hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename="annotations/captions_val2017.json",
        local_dir="data/coco_val2017"
    )
    print("  Annotations downloaded!")

    print("Downloading COCO images (100 subset)...")
    all_files = list(list_repo_files(HF_REPO, repo_type="dataset"))
    img_files = [f for f in all_files if f.startswith("images/")]

    for i, fname in enumerate(img_files):
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset",
            filename=fname,
            local_dir="data/coco_val2017"
        )
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(img_files)} images downloaded")

    print("All images downloaded!")

    # ── Recompute embeddings for deployed subset ───────
    print("Computing embeddings for deployed subset...")
    sys.path.insert(0, "src")

    import torch
    import open_clip
    from torch.utils.data import DataLoader
    from dataset import COCODataset
    from model import VIBEModel

    device   = "cpu"
    ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")

    _, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    dataset    = COCODataset(ann_path, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16,
                            shuffle=False, num_workers=0)

    ckpt  = torch.load("checkpoints/vibe_lean.pt", map_location="cpu")
    model = VIBEModel(device=device, use_projection=True)
    model.ebm.load_state_dict(ckpt["ebm"])
    model.img_proj.load_state_dict(ckpt["img_proj"])
    model.txt_proj.load_state_dict(ckpt["txt_proj"])
    model.eval()

    all_img_embs = []
    all_txt_embs = []

    with torch.no_grad():
        for imgs, txts, _ in dataloader:
            img_emb, txt_emb = model.encode(imgs, txts)
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())

    torch.save({
        "img_embs": torch.cat(all_img_embs),
        "txt_embs": torch.cat(all_txt_embs)
    }, "checkpoints/embeddings.pt")
    print("Embeddings recomputed and saved!")

    # ── Mark setup complete ────────────────────────────
    CACHE_FLAG.touch()
    print("Setup complete!")


if __name__ == "__main__":
    setup()