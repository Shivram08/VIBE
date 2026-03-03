import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from visualize_attention import get_attention_maps, attention_rollout, overlay_attention
from encoders import ImageEncoder


def render(model, tokenizer, device, img_embs, txt_embs, index, img_ids):
    st.title("🔍 Cross-Modal Retrieval")
    st.markdown("Upload an image and retrieve the most semantically aligned captions from the COCO pool using energy scores.")

    # ── Image input ────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        use_sample = st.checkbox("Use a sample image instead", value=True)

        if use_sample and not uploaded:
            from pathlib import Path
            sample_paths = list(Path("data/coco_val2017/images").glob("*.jpg"))
            sample_idx   = st.slider("Sample image index", 0, 50, 0)
            img_path     = sample_paths[sample_idx]
            image        = Image.open(img_path).convert("RGB")
        elif uploaded:
            image = Image.open(uploaded).convert("RGB")
        else:
            st.info("Please upload an image or select a sample.")
            return

        st.image(image, caption="Input Image", use_container_width=True)

    # ── Encode + retrieve ──────────────────────────────────
    with col2:
        st.subheader("Top-5 Retrieved Captions")

        img_enc = ImageEncoder(device)
        img_tensor = img_enc.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            img_emb, _ = model.encode(img_tensor, tokenizer([""]))

        # Compute similarity against all text embeddings
        sim     = (img_emb.cpu() @ txt_embs.T).squeeze()
        top5    = sim.topk(5)
        top_idx = top5.indices.tolist()
        top_sim = top5.values.tolist()

        # Energy scores
        txt_stack = txt_embs[top_idx].to(device)
        with torch.no_grad():
            energies = model.ebm.energy(
                img_emb.expand(5, -1), txt_stack
            ).diag().cpu().tolist()

        # Display results
        results = []
        for rank, (idx, sim_val, energy) in enumerate(zip(top_idx, top_sim, energies)):
            img_id  = img_ids[idx]
            caption = index[img_id]["captions"][0]
            results.append((rank+1, caption, sim_val, energy))
            st.markdown(f"**#{rank+1}** — {caption}")
            st.caption(f"Similarity: {sim_val:.4f} | Energy: {energy:.4f}")
            st.divider()

        # Energy bar chart
        st.subheader("Energy Scores")
        captions_short = [r[1][:40]+"..." for r in results]
        energies_vals  = [r[3] for r in results]

        fig = go.Figure(go.Bar(
            x=energies_vals,
            y=captions_short,
            orientation="h",
            marker_color=["green" if e == min(energies_vals) else "tomato"
                          for e in energies_vals]
        ))
        fig.update_layout(
            title="Energy per Caption (lower = better match)",
            xaxis_title="Energy E(x, y)",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Attention map ──────────────────────────────────────
    st.subheader("ViT Attention Rollout — What the model focuses on")
    attentions = get_attention_maps(img_enc, img_tensor)
    if attentions:
        attn_map = attention_rollout(attentions)
        overlay  = overlay_attention(image, attn_map)
        c1, c2   = st.columns(2)
        c1.image(image.resize((224, 224)),          caption="Original",          use_container_width=True)