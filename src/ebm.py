import torch
import torch.nn as nn


class EBMHead(nn.Module):
    """
    Energy-Based Model head.
    Computes E(x, y) = -dot(fv(x), ft(y)) / τ

    Lower energy = higher compatibility between image and text.
    τ (temperature) is a learnable parameter initialized to 0.07 (CLIP default).
    """
    def __init__(self, init_temp=0.07, min_temp=0.01, max_temp=1.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_temp)))
        self.min_temp = min_temp
        self.max_temp = max_temp

    @property
    def tau(self):
        # Clamp tau to a safe range during training
        return torch.clamp(self.log_tau.exp(), self.min_temp, self.max_temp)

    def energy(self, img_emb, txt_emb):
        """
        Compute pairwise energy matrix E(xi, yj) for all i, j in batch.
        Shape: (B, B) where diagonal = matched pairs (should be low energy)
        """
        return -(img_emb @ txt_emb.T) / self.tau

    def forward(self, img_emb, txt_emb):
        return self.energy(img_emb, txt_emb)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ebm = EBMHead().to(device)

    # Random normalized embeddings (batch of 4)
    img_emb = torch.randn(4, 512).to(device)
    txt_emb = torch.randn(4, 512).to(device)
    img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
    txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)

    E = ebm(img_emb, txt_emb)

    print(f"Energy matrix shape   : {E.shape}")
    print(f"Energy matrix         :\n{E.round(decimals=3)}")
    print(f"Temperature τ         : {ebm.tau.item():.4f}")
    print(f"Diagonal (matched)    : {E.diag().round(decimals=3)}")
    print(f"Mean energy           : {E.mean().item():.4f}")
    print("EBM Head OK!")