import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
import open_clip
# Add after existing imports:
from caption_utils import get_default_captions, format_caption_defaults, parse_caption_input
from dataset import load_coco_index

from encoders import ImageEncoder
from mcmc import run_mcmc
from visualize_attention import get_attention_maps, attention_rollout

import hashlib

def image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()[:10]
# ── Cached resources ───────────────────────────────────────
@st.cache_resource
def get_encoders(device):
    enc = ImageEncoder(device)
    tok = open_clip.get_tokenizer("ViT-B-32")
    return enc, tok

@st.cache_data
def load_embeddings():
    return torch.load("checkpoints/embeddings.pt", map_location="cpu")

@st.cache_data
def load_coco_ids():
    idx = load_coco_index(
        Path("data/coco_val2017/annotations/captions_val2017.json"))
    return list(idx.keys()), idx


def encode_image_and_captions(model, img_enc, tokenizer, image, captions, device):
    """Encode image + captions, return img_emb, txt_embs, dots."""
    img_tensor = img_enc.preprocess(image).unsqueeze(0)
    tokens     = tokenizer(captions)
    with torch.no_grad():
        img_emb, _ = model.encode(img_tensor, tokenizer([""]))
        _, txt_embs = model.encode(
            img_tensor.expand(len(captions), -1, -1, -1), tokens
        )
    dots = (img_emb @ txt_embs.T).squeeze().cpu().detach()
    return img_emb, txt_embs, dots


def render(model):
    st.title("📐 Math Explainer")
    st.markdown("Every chart and metric here is computed live from the **actual trained VIBE model**.")

    device    = model.device
    img_enc, tokenizer = get_encoders(device)
    embeddings         = load_embeddings()
    all_txt_embs       = embeddings["txt_embs"]
    coco_ids, coco_index = load_coco_ids()
    trained_tau        = float(model.ebm.tau.item())

    # ── Image input ────────────────────────────────────────
    st.subheader("🖼 Input Image")
    input_col, cap_col = st.columns([1, 2])

    with input_col:
        upload = st.file_uploader("Upload an image", type=["jpg","jpeg","png"],
                                  key="math_upload")
        use_sample = st.checkbox("Use sample image instead", value=not bool(upload))

        if use_sample or not upload:
            import random
            random.seed(42)
            sample_paths = random.sample(
                list(Path("data/coco_val2017/images").glob("*.jpg")), 6)
            sample_idx = st.select_slider(
                "Pick sample", options=list(range(6)),
                format_func=lambda i: f"Sample {i+1}")
            image = Image.open(sample_paths[sample_idx]).convert("RGB")
        else:
            image = Image.open(upload).convert("RGB")

        st.image(image, caption="Input image", use_container_width=True)

    with cap_col:
        st.markdown("**Candidate captions** (one per line — edit or add your own):")
        st.caption("Top-5 most relevant + 3 distractors auto-selected for your image. Edit freely.")

        # Stable hash based on image content
        img_hash  = hashlib.md5(image.tobytes()).hexdigest()[:10]
        cache_key = f"default_captions_{img_hash}"
        widget_key = f"math_widget_{img_hash}"

        # Generate defaults once per unique image
        if cache_key not in st.session_state:
            img_tensor_tmp = img_enc.preprocess(image).unsqueeze(0)
            with torch.no_grad():
                img_emb_tmp, _ = model.encode(img_tensor_tmp, tokenizer([""]))
            top_caps, bot_caps, _, _ = get_default_captions(
                img_emb_tmp, all_txt_embs, coco_index, coco_ids)
            st.session_state[cache_key] = format_caption_defaults(top_caps, bot_caps)
        
        # Always initialize widget from cache if not already set
        if widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state[cache_key]

        captions_input = st.text_area(
            "Captions",
            height=260,
            key=widget_key
        )
        captions = parse_caption_input(captions_input)


    if len(captions) < 2:
        st.warning("Please enter at least 2 captions.")
        return
    # Encode everything
    with st.spinner("Encoding image and captions..."):
        img_emb, txt_embs, dots = encode_image_and_captions(
            model, img_enc, tokenizer, image, captions, device)
        
    st.divider()

    # ══════════════════════════════════════════════════════
    # SECTION 1 — Temperature τ
    # ══════════════════════════════════════════════════════
    st.header("1 · Temperature τ and the Energy Function")
    st.latex(r"E(x,y)=-\frac{f_v(x)^\top f_t(y)}{\tau}"
             r"\qquad p(y|x)=\frac{e^{-E(x,y)}}{\sum_j e^{-E(x,y_j)}}")

    tau_val = st.slider("Adjust τ", 0.01, 1.0, trained_tau, 0.01,
                        help=f"Trained model τ = {trained_tau:.4f}")

    def compute_probs(dots_tensor, tau):
        energies = (-(dots_tensor / tau)).tolist()
        probs    = F.softmax(-torch.tensor(energies), dim=0).tolist()
        entropy  = float(-sum(p * np.log(p+1e-9) for p in probs))
        return energies, probs, entropy

    energies_trained, probs_trained, entropy_trained = compute_probs(dots, trained_tau)
    energies_user,    probs_user,    entropy_user    = compute_probs(dots, tau_val)
    best_idx   = int(np.argmin(energies_user))
    energy_gap = max(energies_user) - min(energies_user)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature τ",        f"{tau_val:.3f}",
              delta=f"{tau_val - trained_tau:+.3f} vs trained")
    c2.metric("Top-1 Confidence",     f"{max(probs_user):.3f}")
    c3.metric("Distribution Entropy", f"{entropy_user:.3f}",
              delta=f"{entropy_user - entropy_trained:+.3f} vs trained")
    c4.metric("Energy Gap",           f"{energy_gap:.2f}")

    caps_short = [c[:28]+"..." if len(c) > 28 else c for c in captions]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"Trained τ={trained_tau:.3f}",
                                        f"Your τ={tau_val:.3f}"])
    for col, probs_vals in enumerate([probs_trained, probs_user], start=1):
        fig.add_trace(go.Bar(
            x=caps_short, y=probs_vals,
            marker_color=["green" if i == best_idx else "steelblue"
                          for i in range(len(probs_vals))],
            showlegend=False
        ), row=1, col=col)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=120))
    fig.update_xaxes(tickangle=30, tickfont=dict(size=8))
    fig.update_yaxes(title_text="p(y|x)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Green = lowest energy caption. Small τ → confident. Large τ → uncertain.")

    st.divider()

    # ══════════════════════════════════════════════════════
    # SECTION 2 — InfoNCE Loss & Energy Matrix
    # ══════════════════════════════════════════════════════
    st.header("2 · InfoNCE Loss & Energy Matrix")
    st.latex(r"\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}"
             r"\log\frac{e^{-E(x_i,y_i)}}{\sum_{j=1}^{N}e^{-E(x_i,y_j)}}")

    N = st.slider("Batch size N (real embeddings)", 8, 128, 32, step=8)

    with torch.no_grad():
        E_matrix = model.ebm(
            embeddings["img_embs"][:N].to(device),
            embeddings["txt_embs"][:N].to(device)
        ).cpu()

    E_matched   = E_matrix.diag()
    mask        = ~torch.eye(N, dtype=bool)
    E_unmatched = E_matrix[mask]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean E(matched)",      f"{E_matched.mean():.2f}")
    c2.metric("Mean E(unmatched)",    f"{E_unmatched.mean():.2f}")
    c3.metric("Energy Gap",           f"{(E_unmatched.mean()-E_matched.mean()):.2f}")
    c4.metric("log(N) baseline",      f"{np.log(N):.2f}")

    fig2 = go.Figure(go.Heatmap(
        z=E_matrix.numpy(), colorscale="RdYlGn_r",
        colorbar=dict(title="Energy")
    ))
    fig2.update_layout(
        title=f"Energy Matrix ({N}×{N}) — Green diagonal = matched pairs",
        xaxis_title="Text index", yaxis_title="Image index",
        height=420, margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Sharper green diagonal = better separation of matched vs unmatched pairs.")

    st.divider()

    # ══════════════════════════════════════════════════════
    # SECTION 3 — Per-Layer Attention Explorer
    # ══════════════════════════════════════════════════════
    st.header("3 · ViT Attention — Layer by Layer")
    st.latex(r"\text{Attention}(Q,K,V)=\text{softmax}"
             r"\!\left(\frac{QK^\top}{\sqrt{d}}\right)V")

    with st.spinner("Extracting attention maps..."):
        img_tensor   = img_enc.preprocess(image).unsqueeze(0)
        all_attns    = get_attention_maps(img_enc, img_tensor)
        full_rollout = attention_rollout(all_attns)

        # Per-layer single attention maps
        layer_maps = []
        for block in img_enc.encoder.transformer.resblocks:
            attn_list = []
            def make_hook(mod):
                orig = mod.forward
                def patched(*args, **kwargs):
                    kwargs["need_weights"]         = True
                    kwargs["average_attn_weights"] = False
                    out, w = orig(*args, **kwargs)
                    if w is not None:
                        attn_list.append(w.detach().cpu())
                    return out, w
                return patched
            orig_fwd        = block.attn.forward
            block.attn.forward = make_hook(block.attn)
            with torch.no_grad():
                img_enc.encoder(img_tensor.to(device))
            block.attn.forward = orig_fwd
            if attn_list:
                avg  = attn_list[0].mean(dim=1)[0]
                gs   = int((avg.shape[0]-1)**0.5)
                lmap = avg[0, 1:].reshape(gs, gs).numpy()
                lmap = (lmap-lmap.min())/(lmap.max()-lmap.min()+1e-9)
                layer_maps.append(lmap)

    n_layers     = len(layer_maps)
    layer_idx    = st.slider("Transformer layer", 1, n_layers, n_layers)
    show_rollout = st.checkbox("Show full rollout instead", value=False)
    attn_map     = full_rollout if show_rollout else layer_maps[layer_idx-1]

    attn_pil    = Image.fromarray((attn_map*255).astype(np.uint8)).resize(
                      (224,224), Image.BILINEAR)
    attn_arr    = np.array(attn_pil)/255.0
    import matplotlib.pyplot as plt
    heatmap     = plt.cm.jet(attn_arr)[:,:,:3]
    img_arr     = np.array(image.resize((224,224)))/255.0
    overlay     = np.clip(0.5*img_arr + 0.5*heatmap, 0, 1)

    c1, c2, c3  = st.columns(3)
    c1.image(image.resize((224,224)),  caption="Original",          use_container_width=True)
    c2.image(attn_arr,                 caption=f"Layer {layer_idx} attention", use_container_width=True, clamp=True)
    c3.image(overlay,                  caption="Overlay",           use_container_width=True)

    # Attention entropy per layer
    entropies = []
    for lm in layer_maps:
        flat = lm.flatten(); flat = flat/(flat.sum()+1e-9)
        entropies.append(float(-np.sum(flat*np.log(flat+1e-9))))

    fig3 = go.Figure(go.Scatter(
        x=list(range(1, n_layers+1)), y=entropies,
        mode="lines+markers",
        line=dict(color="steelblue", width=2),
        marker=dict(size=8,
                    color=["red" if i+1==layer_idx else "steelblue"
                           for i in range(n_layers)])
    ))
    fig3.add_vline(x=layer_idx, line_dash="dash", line_color="red",
                   annotation_text=f"Layer {layer_idx}")
    fig3.update_layout(
        title="Attention Entropy per Layer — Lower = more focused",
        xaxis_title="Layer", yaxis_title="Entropy",
        height=280, margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Early layers attend broadly (high entropy). Later layers focus on task-relevant patches (low entropy).")

    st.divider()

    # ══════════════════════════════════════════════════════
    # SECTION 4 — Langevin MCMC Live
    # ══════════════════════════════════════════════════════
    st.header("4 · Langevin MCMC — Live on Your Image")
    st.latex(r"y_{t+1}=y_t-\frac{\alpha}{2}\nabla_y E(x,y_t)+\sqrt{\alpha}\,\varepsilon,"
             r"\quad\varepsilon\sim\mathcal{N}(0,I)")

    c1, c2, c3 = st.columns(3)
    n_steps    = c1.slider("Steps", 10, 100, 50)
    step_size  = c2.select_slider("Step size α",
                 [0.0001, 0.001, 0.005, 0.01], value=0.001)
    anneal     = c3.checkbox("Anneal step size", value=True)
    step_sizes_all = [0.0001, 0.001, 0.01]

    if st.button("▶ Run MCMC", type="primary"):
        all_runs = {}
        with st.spinner("Running MCMC..."):
            for alpha in step_sizes_all:
                txt_init   = F.normalize(torch.randn(1,512), dim=-1).to(device)
                trajectory = run_mcmc(img_emb, txt_init, model.ebm,
                                      n_steps=n_steps, step_size=alpha,
                                      anneal=anneal)
                nearest = []
                for step in trajectory:
                    sims = (step["txt_emb"] @ all_txt_embs.T).squeeze()
                    nearest.append(sims.argmax().item())
                all_runs[alpha] = {
                    "energies": [s["energy"] for s in trajectory],
                    "nearest" : nearest
                }
        st.session_state["mcmc_runs"]   = all_runs
        st.session_state["mcmc_steps"]  = n_steps

    if "mcmc_runs" in st.session_state:
        runs     = st.session_state["mcmc_runs"]
        run      = runs[step_size]
        energies = run["energies"]
        nearest  = run["nearest"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Start Energy", f"{energies[0]:.4f}")
        m2.metric("Final Energy", f"{energies[-1]:.4f}")
        m3.metric("Min Energy",   f"{min(energies):.4f}")

        step_view      = st.slider("Explore step", 0, len(energies)-1, 0)
        nearest_cap    = coco_index[coco_ids[nearest[step_view]]]["captions"][0]
        st.info(f"**Step {step_view}** | Energy: `{energies[step_view]:.4f}` | "
                f"Nearest caption: *\"{nearest_cap}\"*")

        fig4 = go.Figure()
        colors = ["tomato", "darkorange", "steelblue"]
        for i, alpha in enumerate(step_sizes_all):
            if alpha in runs:
                fig4.add_trace(go.Scatter(
                    x=list(range(len(runs[alpha]["energies"]))),
                    y=runs[alpha]["energies"],
                    mode="lines", name=f"α={alpha}",
                    line=dict(color=colors[i], width=2)
                ))
        fig4.add_vline(x=step_view, line_dash="dash", line_color="white",
                       annotation_text=f"Step {step_view}")
        fig4.update_layout(
            title="Energy over MCMC Steps — Comparing Step Sizes",
            xaxis_title="Step", yaxis_title="Energy E(x,y)",
            height=320, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Small α = stable but slow. Large α = fast but noisy. Annealing = fast early, stable late.")
    else:
        st.info("Click **▶ Run MCMC** to start the sampler.")

    st.divider()

    # ══════════════════════════════════════════════════════
    # SECTION 5 — EBM ↔ Diffusion Connection
    # ══════════════════════════════════════════════════════
    st.header("5 · EBM ↔ Diffusion Models — The Same Math")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**VIBE — Energy-Based Model**")
        st.latex(r"y_{t+1}=y_t-\frac{\alpha}{2}\nabla_y E(x,y_t)+\sqrt{\alpha}\,\varepsilon")
        st.markdown("""
        - Energy landscape over **text embeddings**
        - Score = **−∇E(x,y)**
        - Sampling conditioned on image x
        """)
    with c2:
        st.markdown("**DDPM — Diffusion Model**")
        st.latex(r"x_{t-1}=\frac{1}{\sqrt{\alpha_t}}"
                 r"\!\left(x_t+(1-\alpha_t)s_\theta(x_t,t)\right)+\sigma_t\varepsilon")
        st.markdown("""
        - Energy landscape over **pixel/latent space**
        - Score = **∇log p(xₜ)**
        - Unconditional or class-conditioned
        """)

    st.success("**Key insight:** Both learn the **gradient of the log probability** (score function). "
               "VIBE's Langevin sampler *is* a conditional diffusion reverse process in text embedding space.")

    # Live score function visualization using input image
    st.subheader("Score Function — Text Embedding Space")
    N_vis   = 100
    txt_sub = all_txt_embs[:N_vis]
    with torch.no_grad():
        E_vals = model.ebm(
            img_emb.expand(N_vis,-1).to(device),
            txt_sub.to(device)
        ).diag().cpu().numpy()

    pca    = PCA(n_components=2)
    txt_2d = pca.fit_transform(txt_sub.numpy())

    fig5 = go.Figure(go.Scatter(
        x=txt_2d[:,0], y=txt_2d[:,1],
        mode="markers",
        marker=dict(size=10, color=E_vals, colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Energy")),
        text=[f"E={e:.2f}" for e in E_vals], hoverinfo="text"
    ))
    fig5.update_layout(
        title="Text embeddings colored by energy for your image (red=high, green=low)",
        xaxis_title="PC1", yaxis_title="PC2",
        height=400, margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("The MCMC sampler follows the gradient from red → green. "
               "This gradient field is the score function — identical in structure to DDPM.")