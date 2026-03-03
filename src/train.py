import sys
sys.path.insert(0, "src")

import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import open_clip

from config import Config
from dataset import COCODataset
from model import VIBEModel
from evaluate import recall_at_k


def train():
    cfg    = Config()
    device = cfg.device
    Path(cfg.ckpt_dir).mkdir(exist_ok=True)

    # ── W&B ────────────────────────────────────────────────
    wandb.init(
        project = cfg.wandb_project,
        entity  = cfg.wandb_entity,
        name    = cfg.run_name,
        config  = cfg.__dict__
    )

    # ── Data ───────────────────────────────────────────────
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer        = open_clip.get_tokenizer("ViT-B-32")

    full_dataset = COCODataset(cfg.ann_path, preprocess, tokenizer)
    val_size     = 500
    train_size   = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)

    # ── Model ──────────────────────────────────────────────
    model     = VIBEModel(device=device, use_projection=cfg.use_projection)
    optimizer = torch.optim.AdamW(model.get_learnable_params(),
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.epochs)

    best_recall = 0.0

    # ── Training Loop ──────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # Keep encoders frozen
        model.img_encoder.encoder.eval()
        model.txt_encoder.encoder.eval()

        total_loss = 0.0
        for imgs, txts, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            optimizer.zero_grad()
            loss, _ = model(imgs, txts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # ── Validation ─────────────────────────────────────
        if epoch % cfg.eval_every == 0:
            r1, r5, r10 = recall_at_k(model, val_loader, device, cfg.recall_ks)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
                  f"R@1: {r1:.3f} | R@5: {r5:.3f} | R@10: {r10:.3f} | "
                  f"τ: {model.ebm.tau.item():.4f}")

            wandb.log({
                "epoch"   : epoch,
                "loss"    : avg_loss,
                "R@1"     : r1,
                "R@5"     : r5,
                "R@10"    : r10,
                "tau"     : model.ebm.tau.item()
            })

            # Save best model
            if r1 > best_recall:
                best_recall = r1
                torch.save(model.state_dict(),
                           Path(cfg.ckpt_dir) / "best_model.pt")
                print(f"  ✓ Best model saved (R@1={r1:.3f})")

    wandb.finish()
    print(f"\nTraining complete! Best R@1: {best_recall:.3f}")


if __name__ == "__main__":
    train()