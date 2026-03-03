import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import open_clip
from encoders import ImageEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_attention_maps(model, image_tensor):
    """
    Extract attention weights by patching MHA modules to return weights.
    """
    attentions = []

    def make_hook(module):
        original_forward = module.forward
        def patched_forward(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, weights = original_forward(*args, **kwargs)
            if weights is not None:
                attentions.append(weights.detach().cpu())
            return out, weights
        return patched_forward

    # Patch all attention modules
    patches = []
    for block in model.encoder.transformer.resblocks:
        original = block.attn.forward
        block.attn.forward = make_hook(block.attn)
        patches.append((block.attn, original))

    with torch.no_grad():
        _ = model.encoder(image_tensor.to(device))

    # Restore original forwards
    for module, original in patches:
        module.forward = original

    return attentions


def attention_rollout(attentions):
    """
    Compute attention rollout across all layers.
    Rolls attention maps from all layers into a single map.
    """
    # Average over heads for each layer
    rollout = torch.eye(attentions[0].shape[-1])

    for attn in attentions:
        attn_avg  = attn.mean(dim=1)[0]          # avg over heads → (seq, seq)
        attn_avg  = attn_avg + torch.eye(attn_avg.shape[0])  # add residual
        attn_avg  = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        rollout   = torch.matmul(attn_avg, rollout)

    # CLS token attends to all patches → row 0
    cls_attn = rollout[0, 1:]   # exclude CLS token itself

    # Reshape to grid (ViT-B/32 → 7x7 patches for 224x224)
    grid_size = int(cls_attn.shape[0] ** 0.5)
    cls_attn  = cls_attn.reshape(grid_size, grid_size).numpy()

    # Normalize
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    return cls_attn


def overlay_attention(image, attn_map):
    """Overlay attention heatmap on original image."""
    img_arr  = np.array(image.resize((224, 224)))
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224),
        Image.BILINEAR)
    ) / 255.0

    # Create heatmap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    overlay = 0.5 * img_arr / 255.0 + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


if __name__ == "__main__":
    img_enc   = ImageEncoder(device)
    img_paths = list(Path("data/coco_val2017/images").glob("*.jpg"))[:6]

    fig, axes = plt.subplots(2, 6, figsize=(20, 7))

    for i, img_path in enumerate(img_paths):
        image        = Image.open(img_path).convert("RGB")
        img_tensor   = img_enc.preprocess(image).unsqueeze(0)
        attentions   = get_attention_maps(img_enc, img_tensor)
        attn_map     = attention_rollout(attentions)
        overlay      = overlay_attention(image, attn_map)

        # Top row: original
        axes[0, i].imshow(image.resize((224, 224)))
        axes[0, i].set_title(f"Image {i+1}", fontsize=8)
        axes[0, i].axis("off")

        # Bottom row: attention overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Attention Rollout", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle("ViT Attention Rollout — What the model focuses on", fontsize=12)
    plt.tight_layout()
    plt.savefig("assets/attention_maps.png", dpi=150)
    plt.show()
    print("Saved to assets/attention_maps.png")