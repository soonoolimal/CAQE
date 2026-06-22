from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class VQCAE(nn.Module):
    def __init__(self, model_cfg: dict, km_n_init: int, km_batch_size: int, km_seed: int):
        super().__init__()

        h_dim = model_cfg["h_dim"]
        vocab_size = model_cfg["vocab_size"]

        e_dim = model_cfg["e_dim"]
        enc_h_dim = model_cfg["enc_h_dim"]
        dec_h_dim = model_cfg["dec_h_dim"]
        clf_h_dim = model_cfg["clf_h_dim"]

        self.n_e = model_cfg["n_e"]
        self.beta = model_cfg["beta"]

        self.encoder = nn.Sequential(
            nn.Linear(h_dim, enc_h_dim),
            nn.LayerNorm(enc_h_dim),
            nn.ReLU(),
            nn.Linear(enc_h_dim, e_dim),
        )
        self.codebook = nn.Embedding(self.n_e, e_dim)
        self.decoder = nn.Sequential(
            nn.Linear(e_dim, dec_h_dim),
            nn.LayerNorm(dec_h_dim),
            nn.ReLU(),
            nn.Linear(dec_h_dim, h_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, clf_h_dim),
            nn.LayerNorm(clf_h_dim),
            nn.ReLU(),
            nn.Linear(clf_h_dim, vocab_size),
        )

        self.km_n_init = km_n_init
        self.km_batch_size = km_batch_size
        self.km_seed = km_seed

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(h)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def quantize(self, z_e: torch.Tensor) -> tuple:
        dists = torch.cdist(z_e, self.codebook.weight) ** 2
        min_indices = torch.argmin(dists, dim=1)  # (N,) nearest codebook index per sample
        z_q_raw = self.codebook(min_indices)      # (N, e_dim)

        cbook_loss = F.mse_loss(z_q_raw, z_e.detach())
        commit_loss = self.beta * F.mse_loss(z_q_raw.detach(), z_e)

        quant_error = dists[torch.arange(len(z_e), device=z_e.device), min_indices].mean()

        # STE: pass gradient from z_q to z_e
        # forward:  z_e + (z_q_raw - z_e) = z_q_raw
        # backward: ∂L/∂(z_q_raw - z_e) = 0 (detached) -> ∂L/∂z_e = ∂L/∂z_q
        z_q = z_e + (z_q_raw - z_e).detach()

        # perplexity: exp(H) of the assignment distribution over this batch
        min_encodings = torch.zeros(len(min_indices), self.n_e, device=z_e.device)  # (N, n_e)
        min_encodings.scatter_(1, min_indices.unsqueeze(1), 1)                      # one-hot: each row indicates which codebook entry the z_e was assigned to
        e_mean = min_encodings.mean(dim=0)
        ppl = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return cbook_loss, commit_loss, quant_error, z_q, min_indices, ppl

    def forward(self, h: torch.Tensor, target_token_ids: torch.Tensor) -> dict:
        z_e = self.encode(h)                                                 # (N, h_dim) -> (N, e_dim)
        cbook_loss, commit_loss, quant_error, z_q, min_indices, ppl = self.quantize(z_e)
        h_hat = self.decode(z_q)                                             # (N, e_dim) -> (N, h_dim)
        recon_loss = F.mse_loss(h_hat, h)

        # h_hat is detached so clf_loss gradient does not flow into the decoder or encoder
        clf_logits = self.classifier(h_hat.detach())  # (N, h_dim) -> (N, vocab_size)
        clf_loss = F.cross_entropy(clf_logits, target_token_ids)
        clf_acc = (clf_logits.argmax(dim=-1) == target_token_ids).float().mean()

        return {
            "loss": cbook_loss + commit_loss + recon_loss + clf_loss,
            "cbook_loss": cbook_loss,
            "commit_loss": commit_loss,
            "recon_loss": recon_loss,
            "clf_loss": clf_loss,
            "clf_acc": clf_acc,
            "quant_error": quant_error,
            "z_e": z_e,
            "z_q": z_q,
            "h_hat": h_hat,
            "min_indices": min_indices,
            "ppl": ppl,
        }

    def init_codebook(self, z_e: torch.Tensor, cache_path=None):
        if cache_path is not None and Path(cache_path).exists():
            centers = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.codebook.weight.data.copy_(centers.to(self.codebook.weight.device))
            tqdm.write(f"Stage 1: loaded K-Means cache from {Path(cache_path).name}")
            return
        self._run_kmeans(z_e, desc="K-Means initialization (stage 1)", cache_path=cache_path)

    def reinit_codebook(self, z_e: torch.Tensor):
        self._run_kmeans(z_e, desc="K-Means reinitialization (stage 2)")

    def _run_kmeans(self, z_e: torch.Tensor, desc: str, cache_path=None):
        data = z_e.detach().cpu().numpy()
        n_samples = len(data)

        if n_samples < self.n_e:
            raise ValueError(f"_run_kmeans requires at least n_e={self.n_e} samples, got {n_samples}")
        if self.km_batch_size < self.n_e:
            raise ValueError(f"km_batch_size={self.km_batch_size} must be >= n_e={self.n_e}")

        best_inertia = float("inf")
        best_centers = None
        best_run = -1

        outer = tqdm(range(self.km_n_init), desc=desc, dynamic_ncols=True)
        for i in outer:
            km = MiniBatchKMeans(n_clusters=self.n_e, n_init=1, batch_size=self.km_batch_size, random_state=self.km_seed + i)
            rng = np.random.default_rng(self.km_seed + i)
            indices = rng.permutation(n_samples)

            inner = tqdm(range(0, n_samples, self.km_batch_size), desc=f"run {i+1}/{self.km_n_init}", leave=False, dynamic_ncols=True)
            for start in inner:
                batch = data[indices[start:start + self.km_batch_size]]
                km.partial_fit(batch)
                inner.set_postfix({"inertia": f"{km.inertia_:.4f}"})

            if km.inertia_ < best_inertia:
                best_inertia = km.inertia_
                best_centers = km.cluster_centers_
                best_run = i + 1
            outer.set_postfix({"best_inertia": f"{best_inertia:.4f}", "best_run": best_run})

        centers = torch.tensor(best_centers, dtype=torch.float32)
        if cache_path is not None:
            torch.save(centers, cache_path)
        self.codebook.weight.data.copy_(centers.to(self.codebook.weight.device))

    def caqe(self, h: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(h)
        _, _, _, z_q, _, _ = self.quantize(z_e)
        h_hat = self.decode(z_q)
        probs = F.softmax(self.classifier(h_hat), dim=-1) # (N, vocab_size)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
