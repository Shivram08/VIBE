import open_clip
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import COCODataset, load_coco_index

ann_path  = Path("data/coco_val2017/annotations/captions_val2017.json")
_, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer        = open_clip.get_tokenizer("ViT-B-32")

dataset    = COCODataset(ann_path, preprocess, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
index      = load_coco_index(ann_path)

# Grab one batch
imgs, txts, ids = next(iter(dataloader))

# Display 4 images with all 5 captions each
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for i, ax in enumerate(axes):
    img_id  = ids[i].item()
    entry   = index[img_id]
    img     = plt.imread(entry["image_path"])
    caption = "\n".join(f"{j+1}. {c}" for j, c in enumerate(entry["captions"]))

    ax.imshow(img)
    ax.set_title(caption, fontsize=7, wrap=True)
    ax.axis("off")

plt.tight_layout()
plt.savefig("assets/smoke_test.png", dpi=150)
plt.show()
print("Smoke test saved to assets/smoke_test.png")