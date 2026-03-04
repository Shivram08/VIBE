"""
Run once on Streamlit Cloud to download weights + data from HuggingFace.
Called automatically from app/main.py on first load.
"""
import os
import sys
from pathlib import Path

HF_REPO    = "shiv0805/VIBE"
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

    # ── Download precomputed embeddings ───────────────
    print("Downloading precomputed embeddings...")
    hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename="embeddings_deploy.pt",
        local_dir="checkpoints"
    )
    import shutil
    shutil.copy("checkpoints/embeddings_deploy.pt",
                "checkpoints/embeddings.pt")
    print("Embeddings ready!")

    # ── Mark setup complete ────────────────────────────
    CACHE_FLAG.touch()
    print("Setup complete!")


if __name__ == "__main__":
    setup()