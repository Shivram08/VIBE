import torch
import torch.nn.functional as F


def langevin_step(img_emb, txt_emb, ebm, step_size=0.01):
    """
    Single Langevin MCMC step in text embedding space.

    Update rule:
        y_{t+1} = y_t - (α/2) * ∇_y E(x, y_t) + √α * ε,  ε ~ N(0, I)

    - Gradient term pushes toward low energy (better alignment)
    - Noise term ensures exploration (prevents collapse)
    """
    txt_emb = txt_emb.detach().requires_grad_(True)

    # Compute energy for matched pair (diagonal of energy matrix)
    E = ebm.energy(img_emb, txt_emb).diag().sum()

    # Compute gradient of energy w.r.t. text embedding
    grad = torch.autograd.grad(E, txt_emb)[0]

    # Langevin update
    noise      = torch.randn_like(txt_emb)
    txt_emb_new = txt_emb - (step_size / 2) * grad + (step_size ** 0.5) * noise

    # Re-normalize to unit sphere (stay on hypersphere)
    txt_emb_new = F.normalize(txt_emb_new.detach(), dim=-1)

    return txt_emb_new, E.item()


def run_mcmc(img_emb, txt_emb_init, ebm, n_steps=50, step_size=0.01, anneal=True):
    """
    Run Langevin MCMC for n_steps starting from txt_emb_init.

    Args:
        img_emb      : image embedding (B, 512)
        txt_emb_init : starting text embedding (B, 512)
        ebm          : EBMHead instance
        n_steps      : number of Langevin steps
        step_size    : initial step size α
        anneal       : linearly anneal step size to 0 over steps

    Returns:
        trajectory   : list of (txt_emb, energy) at each step
    """
    trajectory = []
    txt_emb    = txt_emb_init.clone()

    for t in range(n_steps):
        # Anneal step size linearly
        α = step_size * (1 - t / n_steps) if anneal else step_size

        txt_emb, energy = langevin_step(img_emb, txt_emb, ebm, step_size=α)
        trajectory.append({
            "step"    : t,
            "txt_emb" : txt_emb.detach().cpu(),
            "energy"  : energy
        })

    return trajectory


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    import torch
    from ebm import EBMHead

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ebm    = EBMHead().to(device)

    # Simulate: 1 image embedding, 1 random starting text embedding
    img_emb      = F.normalize(torch.randn(1, 512), dim=-1).to(device)
    txt_emb_init = F.normalize(torch.randn(1, 512), dim=-1).to(device)

    print(f"Initial energy : {ebm.energy(img_emb, txt_emb_init).item():.4f}")

    trajectory = run_mcmc(img_emb, txt_emb_init, ebm, n_steps=50, step_size=0.01)

    energies = [s["energy"] for s in trajectory]
    print(f"Final energy   : {energies[-1]:.4f}")
    print(f"Energy reduced : {energies[0] - energies[-1]:.4f}")
    print(f"Trajectory steps: {len(trajectory)}")

    # Show energy at every 10 steps
    print("\nEnergy over steps:")
    for s in trajectory[::10]:
        print(f"  Step {s['step']:>3} | Energy: {s['energy']:.4f}")

    print("\nMCMC Sampler OK!")