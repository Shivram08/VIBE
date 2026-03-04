import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from setup_deployment import setup
setup()
import streamlit as st

st.set_page_config(
    page_title = "VIBE — Visual-Intent Bridge via Energy Scoring",
    page_icon  = "⚡",
    layout     = "wide"
)

# ── Model loader (cached so it loads once) ─────────────────
@st.cache_resource
def load_model():
    import torch
    from model import VIBEModel
    from encoders import ImageEncoder, TextEncoder
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt   = torch.load("checkpoints/vibe_lean.pt", map_location=device)
    model  = VIBEModel(device=device, use_projection=True)
    model.ebm.load_state_dict(ckpt["ebm"])
    model.img_proj.load_state_dict(ckpt["img_proj"])
    model.txt_proj.load_state_dict(ckpt["txt_proj"])
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, tokenizer, device


@st.cache_resource
def load_embeddings():
    import torch
    from dataset import load_coco_index
    from pathlib import Path

    data     = torch.load("checkpoints/embeddings.pt", map_location="cpu")
    index    = load_coco_index(
        Path("data/coco_val2017/annotations/captions_val2017.json")
    )
    img_ids  = list(index.keys())
    return data["img_embs"], data["txt_embs"], index, img_ids


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.image("assets/vibe_logo.png", use_container_width=True) \
    if __import__("pathlib").Path("assets/vibe_logo.png").exists() else None

st.sidebar.title("⚡ VIBE")
st.sidebar.markdown("""
**Visual-Intent Bridge via Energy Scoring**

A multimodal AI system that scores image-text alignment
using Energy-Based Models and Langevin MCMC dynamics.

---
**Model Stats**
- Architecture: CLIP ViT-B/32 + EBM Head
- Dataset: MS-COCO 5k val
- R@1: 0.528 | R@5: 0.868 | R@10: 0.938
- Energy Gap: 9.77

---
""")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Cross-Modal Retrieval",
     "🌄 EBM Landscape Explorer",
     "📐 Math Explainer"]
)

# ── Page routing ───────────────────────────────────────────
model, tokenizer, device = load_model()
img_embs, txt_embs, index, img_ids = load_embeddings()

if page == "🔍 Cross-Modal Retrieval":
    from tab_retrieval import render
    render(model, tokenizer, device, img_embs, txt_embs, index, img_ids)

elif page == "🌄 EBM Landscape Explorer":
    from tab_landscape import render
    render(model, tokenizer, device, index, img_ids)

elif page == "📐 Math Explainer":
    from tab_math import render
    render(model)