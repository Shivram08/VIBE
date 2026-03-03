import sys
sys.path.insert(0, "src")

import torch
from tqdm import tqdm


@torch.no_grad()
def recall_at_k(model, dataloader, device, ks=(1, 5, 10)):
    """
    Computes image-to-text Recall@K on a dataloader.
    For each image, ranks all candidate texts by energy score.
    Recall@K = fraction of images where correct text is in top-K.
    """
    model.eval()
    all_img_embs = []
    all_txt_embs = []

    for imgs, txts, _ in tqdm(dataloader, desc="Evaluating", leave=False):
        img_emb, txt_emb = model.encode(imgs, txts)
        all_img_embs.append(img_emb.cpu())
        all_txt_embs.append(txt_emb.cpu())

    img_embs = torch.cat(all_img_embs)
    txt_embs = torch.cat(all_txt_embs)

    # Similarity matrix (N x N)
    sim = img_embs @ txt_embs.T
    N   = sim.shape[0]

    recalls = {}
    for k in ks:
        topk    = sim.topk(k, dim=1).indices
        correct = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
        recalls[k] = correct.float().mean().item()

    return recalls[1], recalls[5], recalls[10]
