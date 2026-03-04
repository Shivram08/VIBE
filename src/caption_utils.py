import torch


def get_default_captions(img_emb, all_txt_embs, coco_index, coco_ids,
                         n_top=5, n_bottom=3):
    n_pool   = len(coco_ids)  # clamp to actual pool size
    sim      = (img_emb.cpu() @ all_txt_embs[:n_pool].T).squeeze()
    top_idx  = sim.topk(n_top).indices.tolist()
    bot_idx  = sim.topk(n_bottom, largest=False).indices.tolist()

    top_captions = [coco_index[coco_ids[i]]["captions"][0] for i in top_idx]
    bot_captions = [coco_index[coco_ids[i]]["captions"][0] for i in bot_idx]

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