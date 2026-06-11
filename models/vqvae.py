"""
VQVAE components and model for NLP token representations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, h_dim: int, enc_h_dim: int, e_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(h_dim, enc_h_dim),
            nn.LayerNorm(enc_h_dim),
            nn.ReLU(),
            nn.Linear(enc_h_dim, e_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class Projection(nn.Module):
    def __init__(self, h_dim: int, e_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(h_dim, e_dim),
            nn.LayerNorm(e_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class VectorQuantizer(nn.Module):
    """Discretizes z_e to the nearest codebook entry and computes VQ losses."""
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        ema_gamma: float | None,
        kmeans_n_init: int,
        kmeans_batch_size: int,
        kmeans_seed: int,
    ):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.ema_gamma = ema_gamma
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_seed = kmeans_seed

        self.embedding = nn.Embedding(n_e, e_dim)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(n_e))      # N_k (cluster size)
        self.register_buffer("ema_embed_avg", torch.zeros(n_e, e_dim))  # m_k (embedding sum)

    def warm_start(self, z_e: torch.Tensor):
        """Initializes the codebook with K-Means over z_e = Encoder(h) + Projection(h), then seeds EMA buffers to match."""
        data = z_e.detach().cpu().numpy()
        n_samples = len(data)

        if n_samples < self.n_e:
            raise ValueError(
                f"K-Means warm start requires at least n_e samples: "
                f"got {n_samples}, n_e={self.n_e}"
            )
        if self.kmeans_batch_size < self.n_e:
            raise ValueError(
                f"kmeans_batch_size must be >= n_e for MiniBatchKMeans warm start: "
                f"got kmeans_batch_size={self.kmeans_batch_size}, n_e={self.n_e}"
            )

        best_inertia = float("inf")
        best_centers = None

        outer_pbar = tqdm(range(self.kmeans_n_init), desc="K-Means", dynamic_ncols=True)
        for i in outer_pbar:
            km = MiniBatchKMeans(n_clusters=self.n_e, n_init=1, batch_size=self.kmeans_batch_size, random_state=self.kmeans_seed + i)

            rng = np.random.RandomState(self.kmeans_seed + i)
            indices = rng.permutation(n_samples)

            batch_starts = range(0, n_samples, self.kmeans_batch_size)
            inner_pbar = tqdm(batch_starts, desc=f"  run {i+1}/{self.kmeans_n_init}", leave=False, dynamic_ncols=True)
            for start in inner_pbar:
                batch = data[indices[start:start + self.kmeans_batch_size]]
                km.partial_fit(batch)
                inner_pbar.set_postfix({"inertia": f"{km.inertia_:.4f}"})

            if km.inertia_ < best_inertia:
                best_inertia = km.inertia_
                best_centers = km.cluster_centers_

        centers = torch.tensor(best_centers, dtype=torch.float32).to(self.embedding.weight.device)
        self.embedding.weight.data.copy_(centers)

        # seed EMA buffers so the first training step continues from K-Means, not from zero
        if self.ema_gamma is not None:
            self.ema_embed_avg.copy_(centers)
            self.ema_cluster_size.fill_(1.0)

    def forward(self, z_e: torch.Tensor):
        """Quantizes z to the nearest codebook entry via squared L2 distance.

        Args:
            z_e: Continuous latent, Encoder(h) + Projection(h).
                No permute/reshape needed since input is already 1D in LLM domain, unlike CV domain.

        Returns:
            vq_loss: Codebook loss + Commitment loss.
            z_q: Quantized codebook vector with STE gradient.
            assign: (min_encodings, min_indices) tuple.
                min_encodings: One-hot assignment matrix [N, n_e], used in validation to detect dead codebook vectors.
                min_indices: Nearest codebook index per sample [N, 1].
            ppl: Metric for monitoring codebook usage.
        """
        # squared L2 distances
        d = torch.cdist(z_e, self.embedding.weight) ** 2

        # nearest codebook entry
        min_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_indices.shape[0], self.n_e, device=z_e.device)
        min_encodings.scatter_(1, min_indices, 1)  # one-hot

        # select the codebook vector for each sample via one-hot matmul (equivalent to index lookup)
        z_q = torch.matmul(min_encodings, self.embedding.weight)

        # perplexity (for monitoring codebook usage)
        e_mean = min_encodings.mean(dim=0)
        ppl = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # EMA update (training only, if enabled): tracks codebook usage frequency and moves e_k toward assigned z_e
        if self.training and self.ema_gamma is not None:
            n_k = min_encodings.sum(dim=0)              # [n_e] count per entry in this batch
            embed_sum = min_encodings.T @ z_e.detach()  # [n_e, e_dim] sum of assigned z_e

            self.ema_cluster_size = self.ema_gamma * self.ema_cluster_size + (1 - self.ema_gamma) * n_k
            self.ema_embed_avg = self.ema_gamma * self.ema_embed_avg + (1 - self.ema_gamma) * embed_sum

            # update codebook entries: e_k = m_k / N_k
            self.embedding.weight.data = self.ema_embed_avg / self.ema_cluster_size.unsqueeze(1).clamp(min=1e-5)

        # losses
        codebook_loss = torch.mean((z_q - z_e.detach()) ** 2)
        commitment_loss = self.beta * torch.mean((z_q.detach() - z_e) ** 2)
        vq_loss = codebook_loss + commitment_loss

        # STE: pass gradient from z_q to z_e
        # forward pass: z_e + (z_q - z_e) = z_q
        # backward pass: ∂L/∂(z_q - z_e) = 0 (detached) -> ∂L/∂z_e = ∂L/∂z_q
        z_q = z_e + (z_q - z_e).detach()

        return vq_loss, z_q, (min_encodings, min_indices), ppl


class Decoder(nn.Module):
    """Reconstructs h from z_q."""
    def __init__(self, e_dim: int, dec_h_dim: int, h_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(e_dim, dec_h_dim),
            nn.LayerNorm(dec_h_dim),
            nn.ReLU(),
            nn.Linear(dec_h_dim, h_dim),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


class VQVAE(nn.Module):
    """VQ-VAE for NLP token representations."""
    def __init__(self, h_dim: int, cfg: dict):
        super().__init__()

        e_dim = cfg["e_dim"]
        enc_h_dim = cfg["enc_h_dim"]
        dec_h_dim = cfg["dec_h_dim"]
        n_e = cfg["n_e"]
        beta = cfg["beta"]
        ema_gamma = cfg["ema_gamma"]
        kmeans_n_init = cfg["kmeans_n_init"]
        kmeans_batch_size = cfg["kmeans_batch_size"]
        kmeans_seed = cfg["kmeans_seed"]

        self.projection = Projection(h_dim, e_dim)
        self.encoder = Encoder(h_dim, enc_h_dim, e_dim)
        self.quantizer = VectorQuantizer(n_e, e_dim, beta, ema_gamma, kmeans_n_init, kmeans_batch_size, kmeans_seed)
        self.decoder = Decoder(e_dim, dec_h_dim, h_dim)

    def forward(self, h: torch.Tensor) -> dict:
        # h: hidden vector at [MASK] position for MLM, h_{t-1} for NTP

        # residual connection: z_e = Encoder(h) + Projection(h)
        z_e = self.encoder(h) + self.projection(h)

        # quantize
        vq_loss, z_q, assign, ppl = self.quantizer(z_e)

        # reconstruct h from z_q
        h_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(h_hat, h)

        return {
            "loss": recon_loss + vq_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "h_hat": h_hat,
            "z_e": z_e,
            "z_q": z_q,
            "ppl": ppl,
            "assign": assign,
        }

    def encode_and_quantize(self, h: torch.Tensor) -> torch.Tensor:
        """Encodes h and returns z_q without decoding."""
        z_e = self.encoder(h) + self.projection(h)
        _, z_q, _, _ = self.quantizer(z_e)
        return z_q
