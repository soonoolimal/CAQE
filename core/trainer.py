"""
Trainer class for CAQE model training.
"""

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from models.caqe import CAQE


class Trainer:
    """Manages the full CAQE training lifecycle."""

    def __init__(
        self,
        model: CAQE,
        train_loader,
        val_loader,
        train_cfg: dict,
        device: torch.device,
        run_name: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.device = device
        self.run_name = run_name

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    @torch.no_grad()
    def warm_start(self):
        """Collects encoder outputs from the first warm_start_batches and runs K-Means init."""
        self.model.eval()
        z_e_samples = []

        for i, (hidden, _) in enumerate(self.train_loader):
            if i >= self.cfg["warm_start_batches"]:
                break
            hidden = hidden.to(self.device)
            x = hidden + self.model.vqvae.projection(hidden)
            z_e = self.model.vqvae.encoder(x)
            z_e_samples.append(z_e.cpu())

        z_e_all = torch.cat(z_e_samples, dim=0)
        self.model.vqvae.quantizer.warm_start(z_e_all)
        print(f"Warm start complete: {z_e_all.shape[0]:,} samples → K-Means({self.model.vqvae.quantizer.n_e})")

    def train(self):
        """Runs the full training loop."""
        wandb.init(project=self.cfg["wandb_project"], name=self.run_name, config=self.cfg)
        self.warm_start()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.cfg["n_epochs"] + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics, all_min_encodings, z_e_samples = self._val_epoch(epoch)

            n_dead = self._reinit_dead_codes(all_min_encodings, z_e_samples)
            if n_dead > 0:
                print(f"  → {n_dead} dead codebook vector(s) reinitialized with random z_e samples.")
            self._step_scheduler(val_metrics["loss"])

            wandb.log({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "dead_codes": n_dead,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

            print(
                f"Epoch {epoch:>3} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"perplexity={val_metrics['perplexity']:.1f} | "
                f"dead_codes={n_dead}"
            )

            # early stopping (skipped when early_stopping is false, e.g. cosine scheduler)
            if self.cfg["early_stopping"]:
                if val_metrics["loss"] < best_val_loss - self.cfg["early_stopping_min_delta"]:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg["early_stopping_patience"]:
                        print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience_counter} epochs).")
                        break

        wandb.finish()

    def _build_optimizer(self) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )

    def _build_scheduler(self):
        name = self.cfg["scheduler"]
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg["n_epochs"])
        elif name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.cfg["plateau_mode"],
                patience=self.cfg["plateau_patience"],
            )
        return None

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {"loss": 0.0, "recon_loss": 0.0, "vq_loss": 0.0, "clf_loss": 0.0, "perplexity": 0.0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}")
        for hidden, target in pbar:
            hidden = hidden.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(hidden, target)
            out["loss"].backward()
            self.optimizer.step()

            for k in totals:
                totals[k] += out[k].item()
            n_batches += 1

            pbar.set_postfix({
                "loss":       f"{totals['loss'] / n_batches:.4f}",
                "perplexity": f"{totals['perplexity'] / n_batches:.1f}",
            })

        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Runs validation and collects min_encodings and z_e for dead code reinitialization.

        Returns:
            metrics: averaged loss terms.
            all_min_encodings: [N, n_e] accumulated one-hot assignments across the epoch.
            z_e_samples: encoder outputs collected for potential reinitialization.
        """
        self.model.eval()
        totals = {"loss": 0.0, "recon_loss": 0.0, "vq_loss": 0.0, "clf_loss": 0.0, "perplexity": 0.0}
        n_batches = 0
        all_min_encodings = []
        z_e_samples = []

        pbar = tqdm(self.val_loader, desc=f"Val   epoch {epoch}")
        for hidden, target in pbar:
            hidden = hidden.to(self.device)
            target = target.to(self.device)

            out = self.model(hidden, target)
            z_e_samples.append(out["z_e"].cpu())
            all_min_encodings.append(out["min_encodings"].cpu())

            for k in totals:
                totals[k] += out[k].item()
            n_batches += 1

            pbar.set_postfix({
                "loss":       f"{totals['loss'] / n_batches:.4f}",
                "perplexity": f"{totals['perplexity'] / n_batches:.1f}",
            })

        metrics = {k: v / n_batches for k, v in totals.items()}
        return metrics, torch.cat(all_min_encodings, dim=0), torch.cat(z_e_samples, dim=0)

    def _reinit_dead_codes(self, all_min_encodings: torch.Tensor, z_e_samples: torch.Tensor) -> int:
        """Reinitializes codebook entries that were never assigned during the epoch.

        Returns the number of dead codes reinitialized.
        """
        usage = all_min_encodings.sum(dim=0)  # [n_e], total assignment count per code
        dead = (usage == 0).nonzero(as_tuple=True)[0]

        if len(dead) == 0:
            return 0

        # randomly sampled z_e: dead codes have no convergence history,
        # so a random point from the data distribution is a reasonable fresh start
        idx = torch.randint(0, z_e_samples.shape[0], (len(dead),))
        replacements = z_e_samples[idx].to(self.model.vqvae.quantizer.embedding.weight.device)
        self.model.vqvae.quantizer.embedding.weight.data[dead] = replacements
        return len(dead)

    def _step_scheduler(self, val_loss: float):
        if self.scheduler is None:
            return
        if self.cfg["scheduler"] == "plateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
