"""
Cognitive Alignment Quantization-Based Entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqvae import VQVAE


class CAQE(nn.Module):
    """CAQE = VQVAE + Classifier."""
    def __init__(self, h_dim: int, vocab_size: int, cfg: dict):
        super().__init__()

        self.vqvae = VQVAE(h_dim, cfg["vqvae"])

        # maps decoder output h_hat to logits over vocabulary tokens
        clf_h_dim = cfg["clf"]["clf_h_dim"]
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, clf_h_dim),
            nn.LayerNorm(clf_h_dim),
            nn.ReLU(),
            nn.Linear(clf_h_dim, vocab_size),
        )

    def forward(self, h: torch.Tensor, target_token_ids: torch.Tensor) -> dict:
        """Runs the full CAQE forward pass and returns all loss terms.

        Args:
            h: Pre-computed LLM hidden vectors.
            target_token_ids: Target token ids for the classifier.
                MLM: Masked token.
                NTP: token at timestep t.

        Returns:
            loss: Total loss (vq_loss + recon_loss + clf_loss).
            vq_loss: Codebook loss + Commitment loss.
            recon_loss: MSE(h, h_hat).
            clf_loss: CE(Classifier(h_hat), target_token_ids).
            h_hat: Decoder output (reconstructed h).
            z_e: Continuous encoder output before quantization.
            z_q: Quantized latent vector.
            ppl: Codebook utilization metric.
            assign: (min_encodings, min_indices) tuple.
        """
        out = self.vqvae(h)
        h_hat = out["h_hat"]

        # L_clf updates Classifier only: h_hat is detached so gradient does not flow to Decoder or Encoder
        clf_logits = self.classifier(h_hat.detach())
        clf_loss = F.cross_entropy(clf_logits, target_token_ids)  # -log p_clf(x=target_token_id | h_hat)

        out["clf_loss"] = clf_loss
        out["loss"] = out["loss"] + clf_loss

        return out

    def encode_and_quantize(self, h: torch.Tensor) -> torch.Tensor:
        """Encodes h and returns z_q without decoding."""
        return self.vqvae.encode_and_quantize(h)

    def caqe(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes cognitive alignment quantization-based entropy:
            H(p_clf(x | h_hat)) = -sum_x p_clf(x|h_hat) * log p_clf(x|h_hat) for each input h.

        Used at inference to estimate cognitive alignment.
        Semantically similar tokens map to the same e_k -> same h_hat -> same caqe -> Delta H ~= 0.
        """
        z_q = self.encode_and_quantize(h)
        h_hat = self.vqvae.decoder(z_q)
        probs = F.softmax(self.classifier(h_hat), dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
