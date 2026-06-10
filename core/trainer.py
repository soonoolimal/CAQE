"""
Trainer class for CAQE.
"""

import csv
from pathlib import Path

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from models.caqe import CAQE


class Trainer:
    """Manages the full CAQE training loop."""
    def __init__(
        self,
        model: CAQE,
        train_loader,
        val_loader,
        train_cfg: dict,
        models_cfg: dict,
        device: torch.device,
        run_name: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.models_cfg = models_cfg
        self.device = device
        self.run_name = run_name

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # cosine scheduler runs the full schedule (early stopping is incompatible)
        self.use_early_stopping = (self.cfg["early_stopping"] and self.cfg["scheduler"] != "cosine")

    @torch.no_grad()
    def warm_start(self):
        """Collects Encoder(h) outputs from the first warm_start_batches and runs K-Means init."""
        self.model.eval()
        h_enc_samples = []

        for i, (hidden, _) in enumerate(self.train_loader):
            if i >= self.cfg["warm_start_batches"]:
                break
            hidden = hidden.to(self.device)
            h_enc = self.model.vqvae.encoder(hidden)
            h_enc_samples.append(h_enc.cpu())

        h_enc_all = torch.cat(h_enc_samples, dim=0)
        self.model.vqvae.quantizer.warm_start(h_enc_all)
        print(f"Warm start complete: {h_enc_all.shape[0]:,} samples -> K-Means({self.model.vqvae.quantizer.n_e})")

    def train(self):
        """Runs the full training loop."""
        wandb.init(
            project=self.cfg["wandb_project"],
            name=self.run_name,
            config={**self.models_cfg, **self.cfg},
        )
        csv_path, csv_fields, flat_cfg = self._init_csv()

        self.warm_start()

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        epoch_pbar = tqdm(range(1, self.cfg["n_epochs"] + 1), desc="Epochs")
        for epoch in epoch_pbar:
            train_metrics = self._train_epoch(epoch)
            val_metrics, code_usage = self._val_epoch(epoch)

            n_dead = int((code_usage == 0).sum().item())
            if self.cfg["dead_code_reinit"]:
                self._reinit_dead_codes(code_usage)
            self._step_scheduler(val_metrics["loss"])

            lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "dead_codes": n_dead,
                "lr": lr,
                "epoch": epoch,
            })

            self._log_csv(csv_path, csv_fields, flat_cfg, epoch, lr, train_metrics, val_metrics, n_dead)

            epoch_pbar.set_postfix({
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "ppl": f"{val_metrics['ppl']:.1f}",
                "dead": n_dead,
            })

            if epoch % self.cfg["checkpoint_interval"] == 0:
                self._save_interval_checkpoint(epoch)

            if val_metrics["loss"] < best_val_loss - self.cfg["early_stopping_min_delta"]:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics["loss"])
                patience_counter = 0
            elif self.use_early_stopping:
                patience_counter += 1
                if patience_counter >= self.cfg["early_stopping_patience"]:
                    tqdm.write(f"Early stopping triggered at epoch {epoch} (no improvement for {patience_counter} epochs).")
                    break

        # rename best checkpoint and CSV to include best epoch
        suffix = f"_best{best_epoch}"
        ckpt_dir = Path(self.cfg["checkpoint_dir"]) / self.run_name
        ckpt_path = ckpt_dir / f"{self.run_name}.pt"
        if ckpt_path.exists():
            ckpt_path.rename(ckpt_path.with_stem(f"{self.run_name}{suffix}"))
        if csv_path.exists():
            csv_path.rename(csv_path.with_stem(f"{self.run_name}{suffix}"))

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
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg["n_epochs"],
            )
        elif name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.cfg["plateau_mode"],
                patience=self.cfg["plateau_patience"],
            )
        return None

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {"loss": 0.0, "recon_loss": 0.0, "vq_loss": 0.0, "clf_loss": 0.0, "ppl": 0.0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch}")
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
                "loss": f"{totals['loss'] / n_batches:.4f}",
                "ppl": f"{totals['ppl'] / n_batches:.1f}",
            })

        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[dict, torch.Tensor]:
        """Runs validation and accumulates per-code assignment counts for dead-code monitoring.

        Returns:
            metrics: averaged loss terms.
            code_usage: [n_e] total assignment count per codebook entry.
        """
        self.model.eval()
        totals = {"loss": 0.0, "recon_loss": 0.0, "vq_loss": 0.0, "clf_loss": 0.0, "ppl": 0.0}
        n_batches = 0

        code_usage = torch.zeros(self.model.vqvae.quantizer.n_e)

        pbar = tqdm(self.val_loader, desc=f"Validation epoch {epoch}")
        for hidden, target in pbar:
            hidden = hidden.to(self.device)
            target = target.to(self.device)

            out = self.model(hidden, target)

            min_encodings, _ = out["assign"]
            code_usage += min_encodings.sum(dim=0).cpu()

            for k in totals:
                totals[k] += out[k].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{totals['loss'] / n_batches:.4f}",
                "ppl": f"{totals['ppl'] / n_batches:.1f}",
            })

        metrics = {k: v / n_batches for k, v in totals.items()}
        return metrics, code_usage

    def _reinit_dead_codes(self, code_usage: torch.Tensor):
        """Reinitializes dead codebook entries with the most active vector + gaussian noise.

        Dead codes are set to the most-used codebook vector + small gaussian noise
        so that they re-enter competition without duplicating the active vector exactly.
        """
        dead = (code_usage == 0).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return

        weight = self.model.vqvae.quantizer.embedding.weight
        most_active = code_usage.argmax().item()
        source = weight.data[most_active]

        noise_std = weight.data.std() * 0.01
        for k in dead:
            weight.data[k] = source + torch.randn_like(source) * noise_std

        # seed EMA buffers for reinitialized codes
        # so the next EMA update starts from the reinitialized codebook values, not from the stale near-zero state
        quantizer = self.model.vqvae.quantizer
        if quantizer.ema_gamma is not None:
            quantizer.ema_embed_avg[dead] = weight.data[dead]
            quantizer.ema_cluster_size[dead] = 1.0

        # reset Adam moments for reinitialized codes
        # so the next optimizer step starts fresh, not from the stale gradient history of the dead vectors
        state = self.optimizer.state.get(weight)
        if state:
            state["exp_avg"][dead] = 0
            state["exp_avg_sq"][dead] = 0

    def _ckpt_dir(self) -> Path:
        d = Path(self.cfg["checkpoint_dir"]) / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_checkpoint(self, epoch: int, val_loss: float):
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, self._ckpt_dir() / f"{self.run_name}.pt")

    def _save_interval_checkpoint(self, epoch: int):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, self._ckpt_dir() / f"{self.run_name}_epoch{epoch}.pt")

    def _step_scheduler(self, val_loss: float):
        if self.scheduler is None:
            return
        if self.cfg["scheduler"] == "plateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _init_csv(self) -> tuple[Path, list[str], dict]:
        """Creates the CSV file under logs/ and returns (path, fieldnames, flat_cfg)."""
        flat_cfg = {}
        for k, v in {**self.models_cfg, **self.cfg}.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            flat_cfg[f"{k}.{k2}.{k3}"] = v3
                    else:
                        flat_cfg[f"{k}.{k2}"] = v2
            else:
                flat_cfg[k] = v

        fields = (
            ["epoch", "lr"]
            + [f"train_{k}" for k in ["loss", "recon_loss", "vq_loss", "clf_loss", "ppl"]]
            + [f"val_{k}" for k in ["loss", "recon_loss", "vq_loss", "clf_loss", "ppl"]]
            + ["dead_codes"]
            + list(flat_cfg.keys())
        )

        path = Path("logs") / f"{self.run_name}.csv"
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

        return path, fields, flat_cfg

    def _log_csv(
        self, path: Path, fields: list[str], flat_cfg: dict, epoch: int, lr: float,
        train_metrics: dict, val_metrics: dict, n_dead: int,
    ):
        """Appends one epoch row to the local CSV log."""
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "dead_codes": n_dead,
                **flat_cfg,
            })
