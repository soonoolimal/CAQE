import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class VQCAE(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()

        # set from backbone LLM metadata (h_dim, vocab_size vary per backbone)
        h_dim = model_cfg["h_dim"]
        vocab_size = model_cfg["vocab_size"]

        e_dim = model_cfg["e_dim"]
        enc_h_dim = model_cfg["enc_h_dim"]
        dec_dims = model_cfg["dec_dims"]
        clf_h_dim = model_cfg["clf_h_dim"]

        self.n_e = model_cfg["n_e"]

        self.encoder = nn.Sequential(
            nn.Linear(h_dim, enc_h_dim),
            nn.LayerNorm(enc_h_dim),
            nn.ReLU(),
            nn.Linear(enc_h_dim, e_dim),
        )

        self.codebook = nn.Embedding(self.n_e, e_dim)

        dec_layers = []
        prev_dim = e_dim
        for dim in dec_dims:
            dec_layers += [nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU()]
            prev_dim = dim
        dec_layers.append(nn.Linear(prev_dim, h_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.classifier = nn.Sequential(
            nn.Linear(h_dim, clf_h_dim),
            nn.LayerNorm(clf_h_dim),
            nn.ReLU(),
            nn.Linear(clf_h_dim, vocab_size),
        )

    # h: (N, h_dim) -> (N, e_dim)
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(h)

    # z_q: (N, e_dim) -> (N, h_dim)
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def quantize(self, z_e: torch.Tensor) -> tuple:
        # z_e: (N, e_dim), codebook.weight: (n_e, e_dim)

        # cross-pairwise squared distance matrix between z_e and codes
        # dists[i, j]: squared distance between ith z_e and jth code
        dists = torch.cdist(z_e, self.codebook.weight) ** 2  # (N, n_e)

        min_indices = torch.argmin(dists, dim=1)  # nearest code index per z_e: (N,)
        z_q_raw = self.codebook(min_indices)  # nearest code per z_e: (N, e_dim)

        # mean squared distance between each z_e and its corresponding z_q_raw
        quant_error = ((z_q_raw - z_e) ** 2).sum(dim=-1).mean()

        # STE: pass gradient from z_q to z_e
        # forward:  z_e + (z_q_raw - z_e) = z_q_raw
        # backward: ∂L/∂(z_q_raw - z_e) = 0 (detached) -> ∂L/∂z_e = ∂L/∂z_q
        z_q = z_e + (z_q_raw - z_e).detach()

        return z_q, z_q_raw, min_indices, quant_error

    def forward(self, h: torch.Tensor, target_token_ids: torch.Tensor, beta: float) -> dict:
        z_e = self.encode(h)
        z_q, z_q_raw, min_indices, quant_error = self.quantize(z_e)

        cbook_loss = F.mse_loss(z_q_raw, z_e.detach())
        commit_loss = beta * F.mse_loss(z_q_raw.detach(), z_e)

        h_hat = self.decode(z_q)
        recon_loss = F.mse_loss(h_hat, h)

        logits = self.classifier(h_hat.detach())  # gradient from clf_loss does not flow into the decoder or encoder
        clf_loss = F.cross_entropy(logits, target_token_ids)

        cbook_ppl = self.compute_cbook_ppl(min_indices)
        clf_ppl = torch.exp(clf_loss)
        clf_acc = (logits.argmax(dim=-1) == target_token_ids).float().mean()

        return {
            "loss": cbook_loss + commit_loss + recon_loss + clf_loss,
            "cbook_loss": cbook_loss,
            "commit_loss": commit_loss,
            "recon_loss": recon_loss,
            "clf_loss": clf_loss,
            "cbook_ppl": cbook_ppl,
            "clf_ppl": clf_ppl,
            "clf_acc": clf_acc,
            "quant_error": quant_error,
            "z_e": z_e,
            "min_indices": min_indices,
        }

    def run_kmeans(self, z_e: torch.Tensor, km_n_init: int, km_batch_size: int, km_seed: int, desc: str):
        samples = z_e.detach().cpu().numpy()
        n_samples = len(samples)

        if n_samples < self.n_e:
            raise ValueError(f"run_kmeans requires at least n_e={self.n_e} samples, got {n_samples}")
        if km_batch_size < self.n_e:
            raise ValueError(f"km_batch_size={km_batch_size} must be >= n_e={self.n_e}")

        best_inertia = float("inf")
        best_centers = None
        best_run = -1

        # run K-Means km_n_init times manually to monitor inertia per run (n_init=1 but shuffled per run)
        outer = tqdm(range(km_n_init), desc=desc, dynamic_ncols=True)
        for i in outer:
            km = MiniBatchKMeans(n_clusters=self.n_e, n_init=1, batch_size=km_batch_size, random_state=km_seed + i)
            indices = np.random.default_rng(km_seed + i).permutation(n_samples)

            inner = tqdm(
                range(0, n_samples, km_batch_size), desc=f"Run {i + 1}/{km_n_init}", leave=False, dynamic_ncols=True
            )
            for start in inner:
                batch = samples[indices[start : start + km_batch_size]]
                km.partial_fit(batch)
                inner.set_postfix({"inertia": f"{km.inertia_:.4f}"})  # squared dist to nearest centroid

            if km.inertia_ < best_inertia:
                best_inertia = km.inertia_
                best_centers = km.cluster_centers_
                best_run = i + 1
            outer.set_postfix({"best_inertia": f"{best_inertia:.4f}", "best_run": best_run})

        centers = torch.tensor(best_centers, dtype=torch.float32)
        self.codebook.weight.data.copy_(centers.to(self.codebook.weight.device))

    def compute_cbook_ppl(self, min_indices: torch.Tensor) -> torch.Tensor:
        """Compute exp(H) of the code assignment distribution over this batch."""
        min_encodings = torch.zeros(min_indices.shape[0], self.n_e, device=min_indices.device)  # (N, n_e)

        # convert to one-hot
        min_encodings.scatter_(1, min_indices.unsqueeze(1), 1)  # each row (z_e) has 1 at its assigned code index
        e_mean = min_encodings.mean(dim=0)  # usage probability per code: (n_e,)

        return torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

    # per-token quantized entropy: (N, h_dim) hidden -> (N,) entropy over the code-conditioned vocab distribution
    def compute_caqe(self, h: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(h)
        z_q, _, _, _ = self.quantize(z_e)
        h_hat = self.decode(z_q)
        log_probs = F.log_softmax(self.classifier(h_hat), dim=-1)  # (N, vocab_size)
        return -(log_probs.exp() * log_probs).sum(dim=-1)
