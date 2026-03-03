import json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image


def load_coco_index(ann_path: str | Path) -> dict:
    ann_path = Path(ann_path)
    img_dir  = ann_path.parent.parent / "images"

    with open(ann_path, "r") as f:
        raw = json.load(f)

    id_to_file     = {img["id"]: img["file_name"] for img in raw["images"]}
    id_to_captions = defaultdict(list)
    for ann in raw["annotations"]:
        id_to_captions[ann["image_id"]].append(ann["caption"].strip())

    index = {}
    for img_id, file_name in id_to_file.items():
        index[img_id] = {
            "file_name"  : file_name,
            "image_path" : (img_dir / file_name).resolve(),
            "captions"   : id_to_captions[img_id]
        }
    return index


class COCODataset(Dataset):
    """
    Returns one (image_tensor, caption_tokens, image_id) tuple per sample.
    Each image is paired with one randomly selected caption from its 5.
    """
    def __init__(self, ann_path, preprocess, tokenizer):
        self.index      = load_coco_index(ann_path)
        self.ids        = list(self.index.keys())
        self.preprocess = preprocess
        self.tokenizer  = tokenizer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        entry   = self.index[self.ids[idx]]
        image   = Image.open(entry["image_path"]).convert("RGB")
        caption = entry["captions"][idx % len(entry["captions"])]

        img_tensor = self.preprocess(image)
        txt_tokens = self.tokenizer([caption])[0]

        return img_tensor, txt_tokens, self.ids[idx]


if __name__ == "__main__":
    import open_clip
    from torch.utils.data import DataLoader

    ann_path = Path("data/coco_val2017/annotations/captions_val2017.json")

    # Load CLIP preprocessor and tokenizer
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer        = open_clip.get_tokenizer("ViT-B-32")

    # Build dataset + dataloader
    dataset    = COCODataset(ann_path, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Smoke test — one batch
    imgs, txts, ids = next(iter(dataloader))
    print(f"Dataset size    : {len(dataset)}")
    print(f"Image shape     : {imgs.shape}")
    print(f"Token shape     : {txts.shape}")
    print(f"Sample IDs      : {ids.tolist()}")
    print("COCODataset OK!")