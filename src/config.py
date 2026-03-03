from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    ann_path    : str = "data/coco_val2017/annotations/captions_val2017.json"
    ckpt_dir    : str = "checkpoints"

    # Model
    use_projection : bool = True
    embed_dim      : int  = 512

    # Training
    batch_size  : int   = 64
    epochs      : int   = 10
    lr_ebm      : float = 1e-3
    lr_proj     : float = 1e-4
    weight_decay: float = 0.01
    num_workers : int   = 2

    # Evaluation
    recall_ks   : tuple = (1, 5, 10)
    eval_every  : int   = 1         # evaluate every N epochs

    # W&B
    wandb_project : str = "VIBE"
    wandb_entity  : str = "nekkantishiv-"
    run_name      : str = "vibe-run-01"

    # Device
    device : str = "cuda"


if __name__ == "__main__":
    cfg = Config()
    print(cfg)