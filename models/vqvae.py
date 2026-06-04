"""
VQVAE components and model for NLP token representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class ProjectionModule(nn.Module):
    """Increases the variance of h before encoding via residual connection."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # output dim matches h for residual addition
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # [batch, hidden_dim]


class Encoder(nn.Module):
    """Maps h + Projection(h) to the continuous latent z_e."""
    def __init__(self, hidden_dim: int, encoder_hidden: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, encoder_hidden),     # [batch, hidden_dim] -> [batch, encoder_hidden]
            nn.LayerNorm(encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, embedding_dim),  # [batch, encoder_hidden] -> [batch, embedding_dim]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: h + Projection(h)
        return self.net(x)  # [batch, embedding_dim]


class VectorQuantizer(nn.Module):
    """Discretization bottleneck with STE gradient and entropy regularization."""
    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)  # initialize uniformly in [-1/K, 1/K] following the original VQ-VAE

    def warm_start(self, z_e_samples: torch.Tensor) -> None:
        """Initializes the codebook with K-Means cluster centers.

        Call once before training with a representative sample of encoder outputs
        to prevent early codebook collapse and accelerate convergence.
        """
        z = z_e_samples.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_e, n_init=10, random_state=42)
        kmeans.fit(z)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.embedding.weight.data.copy_(centers.to(self.embedding.weight.device))

    def forward(self, z: torch.Tensor):
        """Quantizes z to the nearest codebook entry via squared L2 distance.

        Args:
            z: continuous encoder output.
                No permute/reshape needed since input is already 1D in LLM domain, unlike CV domain.
                Shape: [batch, embedding_dim].

        Returns:
            vq_loss: codebook loss + commitment loss with entropy penalty.
            z_q: quantized vector with STE gradient.
            perplexity: measures codebook utilization.
            min_encodings: one-hot assignment matrix.
                Used in the trainer to detect never assigned codebook entries.
            min_indices: codebook index per sample.
                Used in the trainer to reinitialize never assigned codebook entries with z_e samples.
        """
        # squared L2 distances
        d = torch.cdist(z, self.embedding.weight) ** 2  # [batch, n_e]

        # nearest codebook entry
        min_indices = torch.argmin(d, dim=1).unsqueeze(1)                             # [batch, 1]
        min_encodings = torch.zeros(min_indices.shape[0], self.n_e, device=z.device)  # [batch, n_e]
        min_encodings.scatter_(1, min_indices, 1)                                     # one-hot

        # select the codebook vector for each sample via one-hot matmul (equivalent to index lookup)
        z_q = torch.matmul(min_encodings, self.embedding.weight)  # [batch, embedding_dim]

        # codebook usage entropy
        e_mean = min_encodings.mean(dim=0)
        codebook_entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
        perplexity = torch.exp(codebook_entropy)

        # codebook loss: moves e_k toward z_e (gradient to codebook)
        codebook_loss = torch.mean((z_q - z.detach())**2) - codebook_entropy
        # commitment loss: keeps z_e near e_k (gradient to encoder)
        commitment_loss = self.beta * torch.mean((z_q.detach() - z)**2)
        vq_loss = codebook_loss + commitment_loss

        # STE: pass gradient from z_q to z_e
        z_q = z + (z_q - z).detach()  # [batch, embedding_dim]

        return vq_loss, z_q, perplexity, min_encodings, min_indices


class Decoder(nn.Module):
    """Maps z_q to logits over the vocabulary for reconstruction."""
    def __init__(self, embedding_dim: int, decoder_hidden: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, vocab_size),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: [batch, embedding_dim]
        return self.net(z_q)  # [batch, vocab_size]


class VQVAE(nn.Module):
    """VQ-VAE for NLP token representations."""
    def __init__(self, hidden_dim: int, vocab_size: int, cfg: dict):
        super().__init__()
        self.projection = ProjectionModule(hidden_dim)
        self.encoder = Encoder(hidden_dim, cfg["encoder_hidden"], cfg["embedding_dim"])
        self.quantizer = VectorQuantizer(cfg["n_embeddings"], cfg["embedding_dim"], cfg["beta"])
        self.decoder = Decoder(cfg["embedding_dim"], cfg["decoder_hidden"], vocab_size)

    def forward(self, h: torch.Tensor, target_token_ids: torch.Tensor) -> dict:
        # h: [batch, hidden_dim], target_token_ids: [batch]

        # residual projection
        z_e = self.encoder(h + self.projection(h))

        # quantize
        vq_loss, z_q, perplexity, min_encodings, min_indices = self.quantizer(z_e)

        # decode
        recon_loss = F.cross_entropy(self.decoder(z_q), target_token_ids)

        return {
            "loss": recon_loss + vq_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "z_e": z_e,
            "z_q": z_q,
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_indices": min_indices,
        }

    def encode_and_quantize(self, h: torch.Tensor) -> torch.Tensor:
        """Encodes h and returns z_q without decoding. Used by CAQE.encode_and_quantize().

        Args:
            h: Pre-computed LLM hidden vectors.
                Shape: [batch, hidden_dim].

        Returns:
            z_q: Quantized latent vector.
                Shape: [batch, embedding_dim].
        """
        z_e = self.encoder(h + self.projection(h))
        _, z_q, _, _, _ = self.quantizer(z_e)
        return z_q
