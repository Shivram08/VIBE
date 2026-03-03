import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from encoders import ImageEncoder
from mcmc import run_mcmc


def render(model, tokenizer, device, index, img_ids):
    st.title("🌄 EBM Landscape Explorer")
    st.markdown("Visualize the energy landscape across candidate captions and watch Langevin MCMC walk toward low-energy regions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input")
        sample_paths = list(Path("data/coco_val2017/images").glob("*.jpg"))
        sample_idx   = st.slider("Sample image", 0, 50, 5)
        img_path     = sample_paths[sample_idx]
        image        = Image.open(img_path).convert("RGB")
        st.image(image, use_container_width=True)

        st.subheader("Candidate Captions")
        st.caption("Edit or add your own captions (one per line)")
        default_captions = """a red double decker bus on a street
a dog playing in the park
a person cooking in the kitchen
a cat sitting on a couch
a group of people at a sports event
a bicycle parked near a building
a bowl of fresh fruit on a table
an airplane flying in the sky"""
        captions_input = st.text_area("Captions", default_captions, height=200)
        captions       = [c.strip() for c in captions_input.strip().split("\n") if c.strip()]

        run_btn = st.button("⚡ Compute Energy Landscape", type="primary")

    with col2:
        if not run_btn:
            st.info("Set your image and captions, then click **Compute Energy Landscape**.")
            return

        # Encode image
        img_enc    = ImageEncoder(device)
        img_tensor = img_enc.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            img_emb, _ = model.encode(img_tensor, tokenizer([""]))

        # Encode captions
        tokens   = tokenizer(captions)
        with torch.no_grad():
            _, txt_embs = model.encode(
                img_tensor.expand(len(captions), -1, -1, -1), tokens
            )

        # Energy scores
        with torch.no_grad():
            energies = model.ebm.energy(
                img_emb.expand(len(captions), -1), txt_embs
            ).diag().cpu().tolist()

        # ── Energy bar chart ───────────────────────────────
        st.subheader("Energy Scores per Caption")
        min_e  = min(energies)
        colors = ["green" if e == min_e else "tomato" for e in energies]
        fig    = go.Figure(go.Bar(
            x=[c[:45]+"..." if len(c) > 45 else c for c in captions],
            y=energies,
            marker_color=colors
        ))
        fig.update_layout(
            xaxis_title="Caption",
            yaxis_title="Energy E(x, y)",
            title="Lower energy = better image-text alignment",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── PCA energy landscape ───────────────────────────
        st.subheader("2D Energy Landscape (PCA)")
        txt_np  = txt_embs.cpu().numpy()
        pca     = PCA(n_components=2)
        txt_2d  = pca.fit_transform(txt_np)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=txt_2d[:, 0], y=txt_2d[:, 1],
            mode="markers+text",
            marker=dict(size=16, color=energies, colorscale="RdYlGn_r",
                        showscale=True, colorbar=dict(title="Energy")),
            text=[c[:25] for c in captions],
            textposition="top center",
            textfont=dict(size=9)
        ))
        fig2.update_layout(
            title="PCA of Text Embeddings — colored by energy",
            xaxis_title="PC1", yaxis_title="PC2",
            height=400, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── MCMC trajectory ────────────────────────────────
        st.subheader("Langevin MCMC Trajectory")
        n_steps   = st.slider("MCMC steps", 10, 100, 50)
        step_size = st.select_slider("Step size α", [0.0001, 0.001, 0.005, 0.01], value=0.001)

        txt_init   = F.normalize(torch.randn(1, 512), dim=-1).to(device)
        trajectory = run_mcmc(img_emb, txt_init, model.ebm,
                              n_steps=n_steps, step_size=step_size)
        traj_energies = [s["energy"] for s in trajectory]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(range(len(traj_energies))), y=traj_energies,
            mode="lines+markers", line=dict(color="darkorange", width=2),
            marker=dict(size=4), name="Energy"
        ))
        fig3.add_hline(y=min(traj_energies), line_dash="dash",
                       line_color="green", annotation_text="Min energy")
        fig3.update_layout(
            title="Energy over Langevin MCMC Steps",
            xaxis_title="Step", yaxis_title="Energy E(x, y)",
            height=350, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig3, use_container_width=True)