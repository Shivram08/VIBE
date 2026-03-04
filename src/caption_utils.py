import torch


def get_default_captions(img_emb, all_txt_embs, coco_index, coco_ids, n_top=5, n_bottom=3):
    """
    Returns top-n relevant + bottom-n distractor captions for an image.

    Args:
        img_emb      : (1, 512) image embedding
        all_txt_embs : (N, 512) all COCO text embeddings
        coco_index   : dict mapping img_id -> {captions, ...}
        coco_ids     : list of img_ids matching all_txt_embs order
        n_top        : number of relevant captions (highest similarity)
        n_bottom     : number of distractor captions (lowest similarity)

    Returns:
        captions     : list of 8 strings (5 relevant + 3 distractors)
        top_indices  : indices of top captions
        bottom_indices: indices of bottom captions
    """
    # Cosine similarity against all captions
    sim      = (img_emb.cpu() @ all_txt_embs.T).squeeze()

    # Top-n most similar (relevant)
    top_idx  = sim.topk(n_top).indices.tolist()

    # Bottom-n least similar (distractors)
    bot_idx  = sim.topk(n_bottom, largest=False).indices.tolist()

    top_captions = [
        coco_index[coco_ids[i]]["captions"][0] for i in top_idx
    ]
    bot_captions = [
        coco_index[coco_ids[i]]["captions"][0] for i in bot_idx
    ]

    return top_captions, bot_captions, top_idx, bot_idx


def format_caption_defaults(top_captions, bot_captions, include_distractors=True):
    """
    Format captions for the text area with a separator comment.
    """
    lines = top_captions
    if include_distractors and bot_captions:
        lines += ["--- distractors (low similarity) ---"]
        lines += bot_captions
    return "\n".join(lines)


def parse_caption_input(raw_text):
    """
    Parse caption text area input, ignoring comment lines.
    """
    return [
        line.strip() for line in raw_text.strip().split("\n")
        if line.strip() and not line.strip().startswith("---")
    ]