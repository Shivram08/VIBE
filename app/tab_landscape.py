import sys
sys.path.insert(0, "src")

import hashlib
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from pathlib import Path
import open_clip

from encoders import ImageEncoder
from model import VIBEModel
from mcmc import run_mcmc
from caption_utils import get_default_captions, format_caption_defaults, parse_caption_input
from dataset import load_coco_index


@st.cache_resource
def get_img_encoder(device):
    return ImageEncoder(device)

@st.cache_data
def load_embeddings():
    return torch.load("checkpoints/embeddings.pt", map_location="cpu")

@st.cache_data
def load_coco_ids():
    idx = load_coco_index(
        Path("data/coco_val2017/annotations/captions_val2017.json"))
    return list(idx.keys()), idx


def render(model, tokenizer, device, index, img_ids):

    st.title("🌄 EBM Landscape Explorer")
    st.markdown("Upload your own image or pick a sample — then explore how the model scores different captions.")

    # ── Instructions ───────────────────────────────────
    with st.expander("🗺 How to use this page", expanded=False):
        st.markdown("""
        1. **Upload your own image** or pick from the sample gallery
        2. **Edit the captions** — top-5 most relevant + 3 distractors are auto-selected
        3. Click **⚡ Compute Energy Landscape** to score all captions
        4. **Energy bar chart** — green bar = most aligned caption, red = least aligned
        5. **2D PCA landscape** — see how captions cluster in embedding space, colored by energy
        6. **MCMC trajectory** — watch Langevin dynamics walk toward low-energy regions
        """)

    st.divider()

    img_enc          = get_img_encoder(device)
    embeddings       = load_embeddings()
    all_txt_embs     = embeddings["txt_embs"]
    coco_ids, coco_index = load_coco_ids()

    # ── Image input ────────────────────────────────────
    input_col, cap_col = st.columns([1, 2])

    with input_col:
        st.subheader("🖼 Image")
        uploaded   = st.file_uploader("Upload your own image",
                                      type=["jpg","jpeg","png"],
                                      key="landscape_upload")
        use_sample = st.checkbox("Use sample image instead",
                                 value=not bool(uploaded),
                                 key="landscape_use_sample")

        if use_sample or not uploaded:
            sample_paths = sorted(Path("data/coco_val2017/images").glob("*.jpg"))
            sample_idx   = st.slider("Sample image", 0, 50, 0,
                                     key="landscape_sample_idx")
            image        = Image.open(sample_paths[sample_idx]).convert("RGB")
        else:
            image = Image.open(uploaded).convert("RGB")

        st.image(image, use_container_width=True)

    # ── Caption input ──────────────────────────────────
    with cap_col:
        st.subheader("📝 Captions")
        st.caption("Top-5 most relevant + 3 distractors auto-selected. Edit or add your own.")

        # Stable hash per image
        img_hash   = hashlib.md5(image.tobytes()).hexdigest()[:10]
        cache_key  = f"landscape_default_{img_hash}"
        widget_key = f"landscape_widget_{img_hash}"

        # Encode image + generate defaults
        img_tensor = img_enc.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            img_emb_tmp, _ = model.encode(img_tensor, tokenizer([""]))

        if cache_key not in st.session_state:
            top_caps, bot_caps, _, _ = get_default_captions(
                img_emb_tmp, all_txt_embs, coco_index, coco_ids,
                n_top=5, n_bottom=3)
            st.session_state[cache_key] = format_caption_defaults(
                top_caps, bot_caps, include_distractors=True)

        if widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state[cache_key]

        captions_input = st.text_area(
            "Captions (one per line)",
            height=260,
            key=widget_key
        )
        captions = parse_caption_input(captions_input)
        run_btn  = st.button("⚡ Compute Energy Landscape", type="primary")

    if not run_btn and f"landscape_results_{img_hash}" not in st.session_state:
        st.info("Select an image and click **⚡ Compute Energy Landscape**.")
        return

    # ── Compute ────────────────────────────────────────
    if run_btn:
        if len(captions) < 2:
            st.warning("Please enter at least 2 captions.")
            return

        with st.spinner("Computing energy landscape..."):
            tokens = tokenizer(captions)
            with torch.no_grad():
                img_emb, _ = model.encode(img_tensor, tokenizer([""]))
                _, txt_embs_live = model.encode(
                    img_tensor.expand(len(captions), -1, -1, -1), tokens)
                energies = model.ebm.energy(
                    img_emb.expand(len(captions), -1),
                    txt_embs_live
                ).diag().cpu().tolist()

        st.session_state[f"landscape_results_{img_hash}"] = {
            "captions"  : captions,
            "energies"  : energies,
            "txt_embs"  : txt_embs_live.cpu(),
            "img_emb"   : img_emb.cpu()
        }

    # ── Display results ────────────────────────────────
    res_key = f"landscape_results_{img_hash}"
    if res_key not in st.session_state:
        return

    res      = st.session_state[res_key]
    captions = res["captions"]
    energies = res["energies"]
    txt_embs_res = res["txt_embs"]
    img_emb  = res["img_emb"].to(device)

    st.divider()

    # ── Energy bar chart ───────────────────────────────
    st.subheader("⚡ Energy Scores per Caption")
    st.caption("Lower energy = better alignment between image and caption.")
    caps_s = [c[:45]+"..." if len(c) > 45 else c for c in captions]
    min_e  = min(energies)
    fig1   = go.Figure(go.Bar(
        x=[c[:45]+"..." if len(c) > 45 else c for c in captions],
        y=energies,
        marker_color=["green" if e == min_e else "tomato" for e in energies]
    ))
    fig1.update_layout(
        title="Caption Energy Scores — lower = better match",
        xaxis_title="Caption",
        yaxis_title="Energy E(x, y)",
        height=360,
        margin=dict(l=10, r=10, t=40, b=120)
    )
    fig1.update_xaxes(tickangle=30, tickfont=dict(size=9))
    st.plotly_chart(fig1, use_container_width=True)

    # ── PCA energy landscape ───────────────────────────
    st.subheader("🗺 2D Energy Landscape (PCA)")
    st.caption("Text embeddings projected to 2D, colored by energy. Green = low energy (good match).")
    from sklearn.decomposition import PCA
    txt_np = txt_embs_res.numpy()
    if len(captions) >= 2:
        n_comp = min(2, len(captions))
        pca    = PCA(n_components=n_comp)
        txt_2d = pca.fit_transform(txt_np)
        if n_comp == 1:
            txt_2d = np.column_stack([txt_2d, np.zeros(len(txt_2d))])

        fig2 = go.Figure(go.Scatter(
            x=txt_2d[:, 0], y=txt_2d[:, 1],
            mode="markers+text",
            marker=dict(size=16, color=energies,
                        colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="Energy")),
            text=[c[:25] for c in captions],
            textposition="top center",
            textfont=dict(size=9)
        ))
        fig2.update_layout(
            title="PCA of Text Embeddings — colored by energy",
            xaxis_title="PC1", yaxis_title="PC2",
            height=420, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── MCMC trajectory ────────────────────────────────
    st.subheader("🔀 Langevin MCMC Trajectory")
    st.caption("Starting from random noise in embedding space, Langevin dynamics walks toward low-energy regions.")

    c1, c2, c3 = st.columns(3)
    n_steps    = c1.slider("Steps", 10, 100, 50, key="landscape_mcmc_steps")
    step_size  = c2.select_slider("Step size α",
                    [0.0001, 0.001, 0.005, 0.01],
                    value=0.001, key="landscape_mcmc_alpha")
    anneal     = c3.checkbox("Anneal step size", value=True,
                             key="landscape_mcmc_anneal")

    if st.button("▶ Run MCMC", type="secondary"):
        step_sizes_all = [0.0001, 0.001, 0.01]
        all_runs = {}
        with st.spinner("Running MCMC..."):
            for alpha in step_sizes_all:
                torch.manual_seed(42)   # ← fixed seed for reproducibility
                txt_init   = F.normalize(torch.randn(1, 512), dim=-1).to(device)
                trajectory = run_mcmc(img_emb, txt_init, model.ebm,
                                    n_steps=n_steps, step_size=alpha,
                                    anneal=anneal)
                all_runs[alpha] = [s["energy"] for s in trajectory]
        st.session_state[f"landscape_mcmc_{img_hash}"] = all_runs

    if f"landscape_mcmc_{img_hash}" in st.session_state:
        runs   = st.session_state[f"landscape_mcmc_{img_hash}"]
        colors = ["tomato", "darkorange", "steelblue"]
        fig3   = go.Figure()
        for i, (alpha, traj) in enumerate(runs.items()):
            fig3.add_trace(go.Scatter(
                x=list(range(len(traj))), y=traj,
                mode="lines", name=f"α={alpha}",
                line=dict(color=colors[i], width=2)
            ))
        fig3.update_layout(
            title="Energy over MCMC Steps — Comparing Step Sizes",
            xaxis_title="Step", yaxis_title="Energy E(x,y)",
            height=340, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig3, use_container_width=True)
        with st.expander("📖 Understanding this chart", expanded=True):
            st.markdown("""
            **What are the controls?**
            - **Steps** — how many Langevin update iterations to run. More steps = longer walk through embedding space
            - **Step size α** — how far to move at each step. Small α = cautious small steps. Large α = big jumps
            - **Anneal step size** — gradually shrinks α from its initial value to 0 over the run. This helps the sampler settle into a low-energy region rather than keep jumping around at the end

            **Reading the chart:**
            - **Y-axis (Energy)** — lower = the text embedding is more aligned with the image
            - **X-axis (Steps)** — each step is one Langevin update
            - **Red (α=0.0001)** — very small steps: cautious, smooth but slow to explore
            - **Orange (α=0.001)** — balanced: moderate speed with reasonable stability
            - **Blue (α=0.01)** — large steps: explores aggressively but oscillates heavily

            **Why does energy start low then go up?**
            The sampler starts from a random point in embedding space. That starting point may happen
            to have a low energy by chance — this is random initialization, not meaningful alignment.
            As the sampler explores, the noise term pushes it around the landscape before it
            eventually settles. With annealing enabled, the final steps are more stable.

            **Why does the energy oscillate?**
            The noise term √α·ε is intentional — it prevents collapse to a single point
            and allows exploration. This is identical to the temperature parameter in DDPM diffusion.
            """)

            st.markdown("**Connection to Diffusion Models:**")
            st.latex(r"y_{t+1} = y_t - \frac{\alpha}{2}\nabla_y E(x, y_t) + \sqrt{\alpha}\,\varepsilon")
            st.markdown("""
            This update is mathematically identical to the DDPM reverse diffusion step.
            Both use a score function — the gradient of log probability — to guide
            sampling toward high-probability (low-energy) regions.
            Head to the **Math Explainer** tab for the full side-by-side derivation.
            """)
    else:
        st.info("Click **▶ Run MCMC** to visualize Langevin dynamics.")