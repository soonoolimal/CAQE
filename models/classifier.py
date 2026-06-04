"""
MLP Classifier: predicts target tokens from the discrete latent z_q.
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Maps z_q to logits over vocabulary tokens."""
    def __init__(self, embedding_dim: int, clf_hidden: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, clf_hidden),
            nn.LayerNorm(clf_hidden),
            nn.ReLU(),
            nn.Linear(clf_hidden, vocab_size),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: [batch, embedding_dim]
        return self.net(z_q)  # [batch, vocab_size]
