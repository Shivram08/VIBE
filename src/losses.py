import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Contrastive InfoNCE loss for image-text matching.

    Given a batch of N (image, text) pairs:
    - Diagonal of energy matrix = matched pairs (should be low energy)
    - Off-diagonal = mismatched pairs (should be high energy)

    Loss = cross entropy over negative energies (= logits)
    Computed symmetrically: image→text + text→image, averaged.

    With random embeddings, expected loss ≈ log(batch_size).
    """
    def __init__(self):
        super().__init__()

    def forward(self, energy_matrix):
        B = energy_matrix.shape[0]

        # Logits = negative energy (higher = more compatible)
        logits = -energy_matrix
        labels = torch.arange(B, device=energy_matrix.device)

        # Image → Text loss: each image finds its matching text
        loss_i2t = F.cross_entropy(logits, labels)

        # Text → Image loss: each text finds its matching image
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


if __name__ == "__main__":
    import math
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = InfoNCELoss()

    # With random embeddings, loss should be ≈ log(batch_size)
    for B in [4, 16, 64, 256]:
        img_emb = F.normalize(torch.randn(B, 512), dim=-1).to(device)
        txt_emb = F.normalize(torch.randn(B, 512), dim=-1).to(device)

        from ebm import EBMHead
        ebm = EBMHead().to(device)
        E   = ebm(img_emb, txt_emb)

        loss     = loss_fn(E)
        expected = math.log(B)
        print(f"B={B:>4} | Loss: {loss.item():.4f} | Expected ≈ log(B): {expected:.4f}")

    print("\nInfoNCE Loss OK!")