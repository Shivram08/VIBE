import sys
sys.path.insert(0, "src")

import torch
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from pathlib import Path

from encoders import ImageEncoder
from dataset import load_coco_index
from visualize_attention import get_attention_maps, attention_rollout, overlay_attention


@st.cache_resource
def get_img_encoder(device):
    return ImageEncoder(device)

@st.cache_data
def load_coco_ids():
    idx = load_coco_index(
        Path("data/coco_val2017/annotations/captions_val2017.json"))
    return list(idx.keys()), idx


def render(model, tokenizer, device, img_embs, txt_embs, index, img_ids):

    # ── Hero ───────────────────────────────────────────
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1a1a2e, #16213e);
                padding: 24px; border-radius: 12px; margin-bottom: 16px;'>
        <h1 style='color: white; margin:0;'>⚡ VIBE</h1>
        <h3 style='color: #a0a0c0; margin: 8px 0;'>
            Visual-Intent Bridge via Energy Scoring
        </h3>
        <p style='color: #c0c0d0; margin: 0;'>
            A multimodal AI system that scores image-text alignment using
            <b>Energy-Based Models</b> and <b>Langevin MCMC dynamics</b>.
            Built on a frozen CLIP ViT-B/32 backbone with a learnable EBM head
            trained with InfoNCE contrastive loss on MS-COCO.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model explanation ──────────────────────────────
    with st.expander("📖 About VIBE — How it works", expanded=False):
        st.markdown("""
        **What is an Energy-Based Model?**
        An EBM assigns a scalar energy score E(x, y) to every image-text pair.
        Lower energy means the image and text are more compatible.
        The model learns to assign low energy to matched pairs and high energy to mismatched ones.

        **The Energy Function:**
        E(x, y) = −fᵥ(x)ᵀ fₜ(y) / τ
        where fᵥ is the image encoder, fₜ is the text encoder, and τ is a learnable temperature.

        **Training:**
        The model is trained with InfoNCE contrastive loss — pulling matched pairs to low energy
        and pushing mismatched pairs to high energy across batches of 256 pairs.

        **Architecture:**
        - Image encoder: frozen CLIP ViT-B/32 → 512d embedding
        - Text encoder: frozen CLIP text transformer → 512d embedding
        - Projection heads: 2-layer MLP (learnable)
        - EBM head: learnable temperature τ
        - Dataset: MS-COCO 2017 validation (5,000 image-caption pairs)

        **Results:**
        R@1: 0.528 | R@5: 0.868 | R@10: 0.938 | Energy Gap: 9.77
        """)

    with st.expander("🗺 How to use this page", expanded=False):
        st.markdown("""
        1. Use the **slider** to browse through validation set images
        2. The model automatically retrieves the **top-5 most aligned captions** from the 5,000-caption COCO pool
        3. **Energy scores** show how compatible each caption is with the image — lower = better match
        4. The **attention map** shows which image patches the ViT focuses on
        5. Head to **EBM Landscape Explorer** to try your own images and captions
        6. Head to **Math Explainer** to understand the theory behind the model
        """)

    st.divider()

    # ── Image selector ─────────────────────────────────
    img_enc      = get_img_encoder(device)
    coco_ids, coco_index = load_coco_ids()
    all_txt_embs = txt_embs
    sample_paths = sorted(Path("data/coco_val2017/images").glob("*.jpg"))

    sample_idx = st.slider("Browse validation images", 0, 99, 0)
    image      = Image.open(sample_paths[sample_idx]).convert("RGB")

    # ── Encode + retrieve ──────────────────────────────
    results_key = f"retrieval_{sample_idx}"
    if results_key not in st.session_state:
        img_tensor = img_enc.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            img_emb, _ = model.encode(img_tensor, tokenizer([""]))

        # Top-5 from COCO pool
        sim      = (img_emb.cpu() @ all_txt_embs.T).squeeze()
        n_pool   = len(coco_ids)  # use actual pool size, not full 5k
        top5_idx = sim[:n_pool].topk(min(5, n_pool)).indices.tolist()
        pool_results = []
        for idx in top5_idx:
            cap   = coco_index[coco_ids[idx]]["captions"][0]
            t_emb = all_txt_embs[idx].unsqueeze(0).to(device)
            with torch.no_grad():
                energy = model.ebm.energy(img_emb, t_emb).item()
            pool_results.append((cap, energy))

        # Attention map
        attentions = get_attention_maps(img_enc, img_tensor)
        attn_map   = attention_rollout(attentions) if attentions else None
        overlay    = overlay_attention(image, attn_map) if attn_map is not None else None

        st.session_state[results_key] = {
            "pool_results": pool_results,
            "overlay"     : overlay,
            "img_emb"     : img_emb
        }

    res          = st.session_state[results_key]
    pool_results = res["pool_results"]
    overlay      = res["overlay"]

    # ── Results layout ─────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.image(image, caption=f"Validation image #{sample_idx+1}",
                 use_container_width=True)
        if overlay is not None:
            st.image(overlay, caption="ViT Attention Rollout",
                     use_container_width=True)

    with right:
        st.subheader("Top-5 Retrieved Captions")
        st.caption("Retrieved from 5,000-caption COCO pool by energy score")

        for rank, (cap, energy) in enumerate(pool_results):
            icon = "🟢" if rank == 0 else "🔵"
            st.markdown(f"{icon} **#{rank+1}** — {cap}")
            st.caption(f"Energy: {energy:.4f}")
            st.divider()

        # Energy bar chart
        caps_s   = [r[0][:45]+"..." if len(r[0]) > 45 else r[0]
                    for r in pool_results]
        energies = [r[1] for r in pool_results]

        fig = go.Figure(go.Bar(
            x=energies, y=caps_s, orientation="h",
            marker_color=["green" if e == min(energies) else "steelblue"
                          for e in energies]
        ))
        fig.update_layout(
            title="Energy Scores — lower = better match",
            xaxis_title="Energy E(x, y)",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)