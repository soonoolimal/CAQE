"""
Cognitive Alignment Quantized Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import Classifier
from models.vqvae import VQVAE


class CAQE(nn.Module):
    """CAQE = VQVAE + Classifier."""
    def __init__(self, hidden_dim: int, vocab_size: int, cfg: dict):
        super().__init__()
        self.vqvae = VQVAE(hidden_dim, vocab_size, cfg)
        self.classifier = Classifier(cfg["embedding_dim"], cfg["clf_hidden"], vocab_size)

    def forward(self, h: torch.Tensor, target_token_ids: torch.Tensor) -> dict:
        """Runs the full CAQE forward pass and returns all loss terms.

        Args:
            h: Pre-computed LLM hidden vectors.
                Shape: [batch, hidden_dim].
            target_token_ids: Target token ids used as both reconstruction and classifier label.
                Shape: [batch].

        Returns:
            loss: total loss (recon + vq + clf).
            recon_loss: cross-entropy between decoder logits and target_token_ids.
            vq_loss: codebook loss + commitment loss with entropy penalty.
            clf_loss: cross-entropy between classifier logits and target_token_ids.
            z_e: continuous encoder output before quantization.
            z_q: quantized latent vector.
            perplexity: measures codebook utilization.
            min_encodings: one-hot assignment matrix, shape [batch, n_e].
            min_indices: codebook index per sample, shape [batch, 1].
        """
        out = self.vqvae(h, target_token_ids)
        z_q = out["z_q"]

        # classifier loss: z_q predicts the same target token via a separate MLP head
        clf_logits = self.classifier(z_q)  # [batch, vocab_size]
        clf_loss = F.cross_entropy(clf_logits, target_token_ids)

        out["clf_loss"] = clf_loss
        out["loss"] = out["loss"] + clf_loss

        return out

    def encode_and_quantize(self, h: torch.Tensor) -> torch.Tensor:
        """Encodes h and returns z_q without decoding. Used internally by caqe_entropy().

        Args:
            h: Pre-computed LLM hidden vectors.
                Shape: [batch, hidden_dim].

        Returns:
            z_q: Quantized latent vector.
                Shape: [batch, embedding_dim].
        """
        return self.vqvae.encode_and_quantize(h)

    def caqe_entropy(self, h: torch.Tensor) -> torch.Tensor:
        """Computes the Shannon entropy of the decoder output distribution over the vocabulary.

        Higher entropy indicates greater prediction uncertainty, interpreted as higher cognitive load for the input token.
        """
        z_q = self.encode_and_quantize(h)
        logits = self.vqvae.decoder(z_q)
        probs = F.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [batch]
