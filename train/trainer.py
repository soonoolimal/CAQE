import copy
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VQCAE
from train.logger import Logger


class Trainer:
    def __init__(
        self,
        model: VQCAE,
        train_cfg: dict,
        model_cfg: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_name: str,
        chunk_dir: Path,
        device: torch.device,
        logger: Logger,
    ):
        self.model = model
        self.device = device

        self.cfg = train_cfg
        self.model_cfg = model_cfg

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger
        self.run_name = run_name
        self._pca_step_interval = train_cfg["reinit_steps"] // train_cfg["pre_reinit_pca_n"]
        self._pca_n_ze = int(model_cfg["n_e"] * train_cfg["pca_ze_ratio"])

        self._km_cfg = {
            "km_n_init": train_cfg["km_n_init"],
            "km_batch_size": train_cfg["km_batch_size"],
            "km_seed": train_cfg["km_seed"],
        }
        self._cbook_cache_path = chunk_dir / (
            f"codebook_ne{model_cfg['n_e']}_s{train_cfg['seed']}_kms{train_cfg['km_seed']}"
            f"_kmni{train_cfg['km_n_init']}_kmbs{train_cfg['km_batch_size']}.pt"
        )

        self._global_step = 0
        self._reinit_steps = train_cfg["reinit_steps"]
        self._reinit_codebook_done = False
        self._noise_reinit_epochs = int(train_cfg["n_epochs"] * train_cfg["noise_reinit_ratio"])

        self.optimizer = optim.Adam(self.model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        self.scheduler = self._build_scheduler()

    @torch.no_grad()
    def _collect_z_e(self, n: int | None = None) -> torch.Tensor:
        """Encode training hidden vectors through the encoder and return z_e.

        n=None: encodes all batches and returns the full z_e.
        Otherwise: encodes just enough batches and returns n randomly sampled z_e.
        """
        buf = []
        total = 0
        for h, _ in self.train_loader:
            z = self.model.encode(h.to(self.device)).cpu()  # (N, e_dim)
            buf.append(z)
            total += z.shape[0]
            if n is not None and total >= n:
                break
        z_e = torch.cat(buf, dim=0)
        if n is None:
            return z_e
        return z_e[torch.randperm(len(z_e))[:n]]

    @torch.no_grad()
    def _init_codebook(self, verbose: bool = True):
        """Run full K-Means initialization over all training hidden vectors with the random encoder (Reinit 1)."""
        self.model.eval()

        if self._cbook_cache_path.exists():
            centers = torch.load(self._cbook_cache_path, map_location="cpu", weights_only=True)
            self.model.codebook.weight.data.copy_(centers.to(self.device))
            if verbose:
                tqdm.write(f"Reinit 1. Loading cache from {self._cbook_cache_path.name}...")
        else:
            if verbose:
                tqdm.write("Reinit 1. Encoding training vectors...")
            z_e_all = self._collect_z_e()
            self.model.run_kmeans(z_e_all, **self._km_cfg, desc="K-Means init (Reinit 1)")
            torch.save(self.model.codebook.weight.data, self._cbook_cache_path)
            del z_e_all

        if verbose:
            tqdm.write(f"Reinit 1 done. Reinit 2 scheduled at step {self._reinit_steps}.")

    @torch.no_grad()
    def _reinit_codebook(self):
        """Run full K-Means reinitialization over all training hidden vectors with the partially trained encoder (Reinit 2)."""
        self.model.eval()

        tqdm.write(f"Reinit 2. Encoding at step {self._global_step}...")  # _global_step == _reinit_steps
        z_e_all = self._collect_z_e()
        self.model.run_kmeans(z_e_all, **self._km_cfg, desc="K-Means reinit (Reinit 2)")
        self._reset_codebook_adam_state()

        tqdm.write("Reinit 2 done.")
        self.model.train()

        del z_e_all
        torch.cuda.empty_cache()

    def _reinit_dead_codes(self, epoch: int, code_usage: torch.Tensor):
        """Reinitialize dead codes per epoch with Gaussian noise centered at the alive entry mean (Reinit 3)."""
        noise_scale = max(self.cfg["noise_init_scale"] * (1 - epoch / (self._noise_reinit_epochs + 1)), 0.0)
        if noise_scale <= 0:
            return

        dead_mask = (code_usage == 0).to(self.device)  # bool (n_e,)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return

        # (n_e, e_dim) -> (n_alive, e_dim).mean(dim=0) -> (e_dim,)
        alive_centroid = self.model.codebook.weight.data[~dead_mask].mean(dim=0)
        noise = torch.randn(n_dead, self.model.codebook.weight.data.shape[1], device=self.device) * noise_scale
        self.model.codebook.weight.data[dead_mask] = alive_centroid + noise

        tqdm.write(f"Reinit 3. {n_dead} dead codes (epoch {epoch}, scale={noise_scale:.4f}).")

    def _diagnose(self):
        """Run one vanilla (partial) epoch after reinit 1 to log quant_error per step.

        Codebook loss is excluded to match pre-reinit training behavior.
        Model is fully restored to random init state afterward so the main training starts from scratch.
        """
        tqdm.write("Diagnostic: Running one vanilla epoch to log quantization error...")
        random_state = copy.deepcopy(self.model.state_dict())
        random_optim = copy.deepcopy(self.optimizer.state_dict())

        self._init_codebook(verbose=False)

        max_diag_steps = int(self._reinit_steps * self.cfg["diag_steps_ratio"])

        self.model.train()
        pbar = tqdm(self.train_loader, desc="Diagnostic epoch", dynamic_ncols=True)
        for step, (h, target) in enumerate(pbar):
            if step >= max_diag_steps:
                break
            h, target = h.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(h, target, self.cfg["beta"])
            (out["loss"] - out["cbook_loss"]).backward()  # exclude cbook_loss (since codebook is not yet reinitialized)
            self.optimizer.step()

            self.logger.log_step(step, out["quant_error"].item())

        self.model.load_state_dict(random_state)
        self.optimizer.load_state_dict(random_optim)
        tqdm.write("Diagnostic complete. Model is restored to random init state.")

    def train(self):
        """Run the full training loop."""
        if self.cfg["diagnose"]:
            self._diagnose()

        self._init_codebook()

        # fix PCA axes from reinit 1 codebook
        self._snapshot_pca("reinit1", fit=True)

        loss_keys = Logger.LOSS_KEYS
        use_early_stopping = self.cfg["early_stopping"] and self.cfg["scheduler"] != "cosine"

        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0

        epoch_pbar = tqdm(range(1, self.cfg["n_epochs"] + 1), desc="Epochs", dynamic_ncols=True)
        for epoch in epoch_pbar:
            train_metrics = self._train_epoch(epoch, loss_keys)
            val_metrics, val_code_usage = self._val_epoch(epoch, loss_keys)

            if self._reinit_codebook_done:
                self._reinit_dead_codes(epoch, val_code_usage)
            self._step_scheduler()

            self.logger.log_epoch(epoch, train_metrics, val_metrics)

            summary = self._summarize_epoch(train_metrics, val_metrics)
            epoch_pbar.set_postfix(summary)

            is_best = val_metrics["loss"] < best_val_loss - self.cfg["min_delta"]
            if is_best:
                suffix = " (best)"
            elif use_early_stopping:
                suffix = f" (patience {patience + 1}/{self.cfg['patience']})"
            else:
                suffix = ""
            tqdm.write(f"Epoch {epoch} | " + "  ".join(f"{k}={v}" for k, v in summary.items()) + "  " + suffix)

            if epoch % self.cfg["ckpt_save_interval"] == 0:
                self._save_checkpoint(epoch, interval=True)

            if is_best:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics["loss"])
                patience = 0
            elif use_early_stopping:
                patience += 1
                if patience >= self.cfg["patience"]:
                    tqdm.write(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        ckpt_suffix = f"_best{best_epoch}"
        self._rename_checkpoint(ckpt_suffix)
        self.logger.finish()

    def _train_epoch(self, epoch: int, loss_keys: list[str]) -> dict:
        self.model.train()

        totals = dict.fromkeys(loss_keys, 0.0)
        code_usage = torch.zeros(self.model.n_e)

        train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False, dynamic_ncols=True)
        n_batches = 0  # count actual batches, since loader __len__() is approximate for uneven chunk sizes
        for i, (h, target) in enumerate(train_loader):
            if not self._reinit_codebook_done:
                if self._global_step >= self._reinit_steps:
                    train_loader.refresh()
                    self._reinit_codebook()
                    self._reinit_codebook_done = True

                    # post-reinit snapshot to verify codebook alignment
                    self._snapshot_pca("reinit2")
                elif self._global_step > 0 and self._global_step % self._pca_step_interval == 0:
                    # pre-reinit PCA snapshots evenly spaced in (0, reinit_steps)
                    self._snapshot_pca(self._global_step)

            h, target = h.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(h, target, self.cfg["beta"])

            # exclude cbook_loss before reinit so the codebook stays at its K-Means init, only the encoder is trained
            # after reinit 2, the full loss is used
            # validation does not need this adjustment since no backward is performed
            if not self._reinit_codebook_done:
                out["loss"] -= out["cbook_loss"]
            loss = out["loss"]

            loss.backward()
            self.optimizer.step()
            code_usage += torch.bincount(out["min_indices"].detach().cpu(), minlength=self.model.n_e)

            self._global_step += 1

            for k in loss_keys:
                totals[k] += out[k].item()
            train_loader.set_postfix(self._summarize_batch(totals, i))
            n_batches += 1

        metrics = {k: v / n_batches for k, v in totals.items()}

        # recompute ppl = exp(H) from epoch-level
        # since H is the entropy of the distribution which differs with large variance per batch,
        # mean(H_i) != H of the full epoch distribution
        e_mean = code_usage / code_usage.sum()
        metrics["cbook_ppl"] = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10))).item()
        metrics["clf_ppl"] = math.exp(metrics["clf_loss"])

        metrics["n_dead"] = int((code_usage == 0).sum().item())

        return metrics

    @torch.no_grad()
    def _val_epoch(self, epoch: int, loss_keys: list[str]) -> tuple[dict, torch.Tensor]:
        self.model.eval()

        totals = dict.fromkeys(loss_keys, 0.0)
        code_usage = torch.zeros(self.model.n_e)

        val_loader = tqdm(self.val_loader, desc=f"Epoch {epoch} val", leave=False, dynamic_ncols=True)
        n_batches = 0
        for i, (h, target) in enumerate(val_loader):
            h, target = h.to(self.device), target.to(self.device)

            out = self.model(h, target, self.cfg["beta"])
            code_usage += torch.bincount(out["min_indices"].cpu(), minlength=self.model.n_e)

            for k in loss_keys:
                totals[k] += out[k].item()
            val_loader.set_postfix(self._summarize_batch(totals, i))
            n_batches += 1

        metrics = {k: v / n_batches for k, v in totals.items()}

        e_mean = code_usage / code_usage.sum()
        metrics["cbook_ppl"] = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10))).item()
        metrics["clf_ppl"] = math.exp(metrics["clf_loss"])

        metrics["n_dead"] = int((code_usage == 0).sum().item())

        return metrics, code_usage

    def _snapshot_pca(self, tag: str | int, fit: bool = False):
        """Log a PCA snapshot.

        Called 1 + (pre_reinit_pca_n - 1) + 1 times total:
            - Reinit 1 (fit=True)
            - pre_reinit_pca_n - 1 evenly-spaced pre-reinit steps
            - Reinit 2
        """
        if not self.logger.pca_wandb:
            return
        self.model.eval()
        z_e = self._collect_z_e(n=self._pca_n_ze)
        with torch.no_grad():
            dists = torch.cdist(z_e.to(self.device), self.model.codebook.weight.data)  # (N, n_e)
            assignments = dists.argmin(dim=1)  # (N,)
            active_mask = torch.zeros(self.model.codebook.weight.data.shape[0], dtype=torch.bool)  # (n_e,)
            active_mask[assignments.cpu()] = True
        self.model.train()
        if fit:
            self.logger.fit_pca(self.model.codebook.weight.data)
        self.logger.log_pca(tag, z_e, self.model.codebook.weight.data, active_mask=active_mask)

    def _reset_codebook_adam_state(self):
        """Reset Adam moments for the codebook parameter to remove stale momentum after reinit."""
        state = self.optimizer.state.get(self.model.codebook.weight)
        if state:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()

    def _build_scheduler(self):
        if self.cfg["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg["n_epochs"])
        return None

    def _step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def _summarize_batch(self, totals: dict, i: int) -> dict[str, str]:
        n = i + 1
        return {
            "loss": f"{totals['loss'] / n:.4f}",
            "cbook_ppl": f"{totals['cbook_ppl'] / n:.1f}",
            "clf_ppl": f"{totals['clf_ppl'] / n:.1f}",
            "clf_acc": f"{totals['clf_acc'] / n:.3f}",
        }

    def _summarize_epoch(self, train_metrics: dict, val_metrics: dict) -> dict[str, str]:
        return {
            "train_loss": f"{train_metrics['loss']:.4f}",
            "val_loss": f"{val_metrics['loss']:.4f}",
            "val_cbook_ppl": f"{val_metrics['cbook_ppl']:.1f}",
            "val_clf_ppl": f"{val_metrics['clf_ppl']:.1f}",
            "val_clf_acc": f"{val_metrics['clf_acc']:.3f}",
            "val_dead": f"{val_metrics['n_dead']}/{self.model_cfg['n_e']}",
        }

    def _save_checkpoint(self, epoch: int, val_loss: float | None = None, *, interval: bool = False):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if interval:
            path = self.logger.ckpt_dir / f"{self.run_name}_epoch{epoch}.pt"
        else:
            state["val_loss"] = val_loss
            path = self.logger.ckpt_dir / f"{self.run_name}.pt"
        torch.save(state, path)

    def _rename_checkpoint(self, suffix: str):
        ckpt_path = self.logger.ckpt_dir / f"{self.run_name}.pt"
        if ckpt_path.exists():
            ckpt_path.rename(ckpt_path.with_stem(f"{self.run_name}{suffix}"))
