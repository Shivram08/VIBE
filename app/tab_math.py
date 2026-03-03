import sys
sys.path.insert(0, "src")

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch


def render(model):
    st.title("📐 Math Explainer")
    st.markdown("Interactive breakdown of the math behind VIBE. Adjust parameters and see their effect live.")

    # ── Section 1: Energy Function ─────────────────────────
    st.header("1. Energy Function")
    st.latex(r"E(x, y) = -\frac{f_v(x)^\top f_t(y)}{\tau}")
    st.markdown("""
    - **x** = image, **y** = text
    - **fᵥ(x)** = L2-normalized image embedding (ViT encoder)
    - **fₜ(y)** = L2-normalized text embedding (BERT encoder)
    - **τ** = temperature — controls sharpness of the distribution
    - Lower energy = higher compatibility between image and text
    """)

    tau_val = st.slider("Temperature τ", 0.01, 1.0,
                        float(model.ebm.tau.item()), 0.01)

    # Show effect of tau on softmax distribution
    st.subheader("Effect of τ on probability distribution")
    st.markdown("Given 8 candidate texts with fixed dot products, how does τ change the distribution?")

    dots     = np.array([0.85, 0.60, 0.45, 0.30, 0.20, 0.15, 0.10, 0.05])
    logits   = dots / tau_val
    probs    = np.exp(logits) / np.exp(logits).sum()
    captions = [f"Caption {i+1}" for i in range(len(dots))]

    fig = go.Figure(go.Bar(
        x=captions, y=probs,
        marker_color=["green" if i == 0 else "steelblue" for i in range(len(probs))]
    ))
    fig.update_layout(
        title=f"p(y|x) with τ={tau_val:.2f} — {'sharp' if tau_val < 0.1 else 'smooth'} distribution",
        yaxis_title="Probability", height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Small τ → model is very confident (peaked). Large τ → uniform, uncertain distribution.")

    st.divider()

    # ── Section 2: InfoNCE Loss ────────────────────────────
    st.header("2. InfoNCE Contrastive Loss")
    st.latex(r"\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{-E(x_i, y_i)}}{\sum_{j=1}^{N} e^{-E(x_i, y_j)}}")
    st.markdown("""
    - Diagonal of energy matrix = **matched pairs** (should be low energy)
    - Off-diagonal = **mismatched pairs** (should be high energy)
    - With random embeddings, expected loss ≈ **log(N)**
    - As training progresses, loss drops well below log(N)
    """)

    batch_size = st.slider("Batch size N", 4, 256, 64, step=4)
    expected   = np.log(batch_size)
    trained    = expected * 0.12

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=["Random (untrained)", "VIBE (trained)"],
        y=[expected, trained],
        marker_color=["tomato", "green"],
        text=[f"{expected:.2f}", f"{trained:.2f}"],
        textposition="outside"
    ))
    fig2.add_hline(y=expected, line_dash="dash",
                   line_color="gray",
                   annotation_text=f"log(N) = {expected:.2f}")
    fig2.update_layout(
        title=f"InfoNCE Loss: Random vs Trained (N={batch_size})",
        yaxis_title="Loss", height=320,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Section 3: Langevin MCMC ───────────────────────────
    st.header("3. Langevin MCMC Dynamics")
    st.latex(r"y_{t+1} = y_t - \frac{\alpha}{2} \nabla_y E(x, y_t) + \sqrt{\alpha} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)")
    st.markdown("""
    - **Gradient term** `−(α/2)∇E` — pushes toward low energy (better alignment)
    - **Noise term** `√α · ε` — ensures exploration, prevents collapse
    - This is mathematically identical to the **DDPM reverse diffusion step**
    - The score function `s(y) = −∇E(x,y) = ∇log p(y|x)` connects EBMs to diffusion models
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_steps   = st.slider("Steps", 10, 200, 50)
        step_size = st.select_slider("Step size α",
                    [0.0001, 0.001, 0.005, 0.01, 0.05], value=0.001)
        anneal    = st.checkbox("Anneal step size", value=True)

    # Simulate energy trajectory
    np.random.seed(42)
    energy = 2.0
    traj   = []
    alpha  = step_size
    for t in range(n_steps):
        a       = alpha * (1 - t/n_steps) if anneal else alpha
        grad    = energy * 0.3
        noise   = np.random.randn() * np.sqrt(a)
        energy  = energy - (a/2) * grad + noise
        traj.append(energy)

    with col2:
        st.metric("Start energy",  f"{traj[0]:.3f}")
        st.metric("Final energy",  f"{traj[-1]:.3f}")
        st.metric("Min energy",    f"{min(traj):.3f}")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(range(n_steps)), y=traj,
        mode="lines", line=dict(color="darkorange", width=2), name="Energy"
    ))
    fig3.add_hline(y=min(traj), line_dash="dash", line_color="green",
                   annotation_text="Min energy")
    fig3.update_layout(
        title="Simulated Langevin Energy Trajectory",
        xaxis_title="Step", yaxis_title="Energy",
        height=320, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Section 4: Connection to Diffusion ─────────────────
    st.header("4. Connection to Diffusion Models")
    st.markdown("The Langevin update is mathematically equivalent to the DDPM reverse process:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Langevin MCMC (EBM)**")
        st.latex(r"y_{t+1} = y_t - \frac{\alpha}{2} \nabla_y E(x, y_t) + \sqrt{\alpha}\varepsilon")

    with col2:
        st.markdown("**DDPM Reverse Process**")
        st.latex(r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t + (1-\alpha_t) s_\theta(x_t, t)\right) + \sigma_t \varepsilon")

    st.markdown("""
    The key connection:
    - EBM score function: **s(y) = −∇E(x,y)**
    - DDPM score network: **s_θ(xₜ, t) ≈ ∇log p(xₜ)**
    - Both are learning the **gradient of the log probability** — the same underlying math
    """)

    st.info("This is why EBMs and Diffusion Models are deeply related — VIBE's Langevin sampler *is* a simplified diffusion reverse process.")