import copy
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from core.logger import Logger
from core.model import VQCAE


class Trainer:
    def __init__(
        self,
        model: VQCAE,
        train_loader,
        val_loader,
        train_cfg: dict,
        model_cfg: dict,
        device: torch.device,
        run_name: str,
        chunk_dir: Path,
    ):
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = device
        self.run_name = run_name
        self._initialized_codebook_cache = chunk_dir / (
            f"kmeans_s{train_cfg['seed']}_ne{model_cfg['n_e']}_kms{train_cfg['km_seed']}"
            f"_ni{train_cfg['km_n_init']}_bs{train_cfg['km_batch_size']}.pt"
        )

        self._global_step = 0
        self._reinit_done = False
        self._reinit_step = 0  # set during _init_codebook()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = self._build_scheduler()
        self.use_early_stopping = self.train_cfg["early_stopping"] and self.train_cfg["scheduler"] != "cosine"

        self.logger: Logger | None = None

    @torch.no_grad()
    def _encode_all(self) -> torch.Tensor:
        """Passes all train hidden vectors through the encoder and returns z_e."""
        z_e_buf = []
        for hidden, _ in self.train_loader:
            z_e_buf.append(self.model.encode(hidden.to(self.device)).cpu())
        return torch.cat(z_e_buf, dim=0)  # (N, e_dim)

    @torch.no_grad()
    def _sample_z_e(self, n: int) -> torch.Tensor:
        """Encodes just enough train batches to collect n z_e samples, randomly subsampled."""
        buf = []
        for hidden, _ in self.train_loader:
            buf.append(self.model.encode(hidden.to(self.device)).cpu())
            if sum(z.shape[0] for z in buf) >= n:
                break
        z_e = torch.cat(buf, dim=0)
        return z_e[torch.randperm(len(z_e))[:n]]

    @torch.no_grad()
    def _init_codebook(self):
        """Stage 1: Full K-Means initialization over all train hidden vectors with the random encoder."""
        self.model.eval()
        self._reinit_step = self.train_cfg["reinit_steps"]
        cache_path = self._initialized_codebook_cache

        cb_before = self.model.codebook.weight.data.clone()
        self.logger.fit_pca(cb_before)  # PCA axes fixed on random init before stage 1 K-Means

        if cache_path.exists():
            self.model.init_codebook(None, cache_path=cache_path)
        else:
            tqdm.write("Stage 1: Encoding all train hidden vectors to codebook vectors for K-Means initialization...")
            z_e_all = self._encode_all()
            self.model.init_codebook(z_e_all, cache_path=cache_path)
            del z_e_all

        z_e_sample = self._sample_z_e(Logger.PCA_ZE_SAMPLES)
        self.logger.log_pca("Before Stage 1", cb_before, step=0, z_e=z_e_sample, show_init=True)
        self.logger.log_pca("After Stage 1", self.model.codebook.weight.data, step=0, z_e=z_e_sample, show_init=True, s1_codebook=self.model.codebook.weight.data)
        self.logger.log_codebook_init(self.model.codebook.weight.data)
        tqdm.write(f"Stage 1 complete: reinitialization (stage 2) runs at step {self._reinit_step}.")

    @torch.no_grad()
    def _reinit_codebook(self):
        """Stage 2: Full K-Means reinitialization over all train hidden vectors with the partially trained encoder."""
        self.model.eval()
        tqdm.write(f"Stage 2: Encoding all train hidden vectors for reinitialization at step {self._global_step}...")

        z_e_all = self._encode_all()
        idx = torch.randperm(len(z_e_all))[:Logger.PCA_ZE_SAMPLES]
        z_e_sample = z_e_all[idx]

        s1_cb = self.model.codebook.weight.data.clone()
        self.logger.log_pca("Before Stage 2", self.model.codebook.weight.data, step=self._global_step, z_e=z_e_sample, show_init=True, s1_codebook=s1_cb, subtitle="z_e from trained encoder")

        self.model.reinit_codebook(z_e_all)
        self._reset_codebook_adam_state()
        self.model.train()

        self.logger.log_codebook_reinit(self.model.codebook.weight.data)
        self.logger.log_pca("After Stage 2", self.model.codebook.weight.data, step=self._global_step, z_e=z_e_sample, s1_codebook=s1_cb, s2_codebook=self.model.codebook.weight.data)
        tqdm.write(f"Stage 2 complete: reinitialization done at step {self._global_step}.")

        del z_e_all, z_e_sample
        torch.cuda.empty_cache()

    def _reinit_dead_codes(self, epoch: int, code_usage: torch.Tensor, noise_scale: float):
        """Stage 3: Per-epoch dead code reinitialization with Gaussian noise centered at the alive entry mean."""
        if noise_scale <= 0:
            return

        dead_mask = (code_usage == 0).to(self.device)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return

        self.logger.log_pca(f"Before Stage 3 (Epoch {epoch})", self.model.codebook.weight.data, step=self._global_step, code_usage=code_usage)

        alive_mean = self.model.codebook.weight.data[~dead_mask].mean(dim=0)
        noise = torch.randn(n_dead, self.model.codebook.weight.data.shape[1], device=self.device) * noise_scale
        self.model.codebook.weight.data[dead_mask] = alive_mean + noise

        self.logger.log_pca(f"After Stage 3 (Epoch {epoch})", self.model.codebook.weight.data, step=self._global_step, code_usage=code_usage)
        tqdm.write(f"\tStage 3 (epoch {epoch}): reinitialized {n_dead} dead codes (noise_scale={noise_scale:.4f})")

    def _run_diagnostic(self):
        """Runs one vanilla epoch after Stage 1 init to log quant_error per step.
        All losses are active (no pre-reinit adjustment). Model is fully restored
        to random init state afterward so the main training starts from scratch."""
        tqdm.write("Diagnostic: running one vanilla epoch to log quantization error...")
        random_state = copy.deepcopy(self.model.state_dict())
        random_optim = copy.deepcopy(self.optimizer.state_dict())

        # load Stage 1 codebook without logging
        self.model.eval()
        cache_path = self._initialized_codebook_cache
        if cache_path.exists():
            self.model.init_codebook(None, cache_path=cache_path)
        else:
            z_e_all = self._encode_all()
            self.model.init_codebook(z_e_all, cache_path=cache_path)
            del z_e_all

        self.model.train()
        pbar = tqdm(self.train_loader, desc="Diagnostic epoch", dynamic_ncols=True)
        for step, (hidden, target) in enumerate(pbar):
            hidden = hidden.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(hidden, target)
            (out["loss"] - out["cbook_loss"]).backward()
            self.optimizer.step()

            self.logger.log_diagnostic_step(out["quant_error"].item(), step)

        self.model.load_state_dict(random_state)
        self.optimizer.load_state_dict(random_optim)
        tqdm.write("Diagnostic complete. Model restored to random init state.")

    def train(self):
        """Runs the full training loop."""
        self.logger = Logger(
            project=self.train_cfg["wandb_project"],
            run_name=self.run_name,
            config={**self.model_cfg, **self.train_cfg},
        )

        if self.train_cfg["run_diagnostic"]:
            self._run_diagnostic()
        self._init_codebook()
        self.logger.update_config({"reinit_step": self._reinit_step})

        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0

        noise_reinit_epochs = int(self.train_cfg["n_epochs"] * self.train_cfg["noise_reinit_ratio"])
        epoch_pbar = tqdm(range(1, self.train_cfg["n_epochs"] + 1), desc="Epochs", dynamic_ncols=True)
        for epoch in epoch_pbar:
            train_metrics, train_code_usage = self._train_epoch(epoch)
            val_metrics, val_code_usage = self._val_epoch(epoch)

            e_mean = train_code_usage / train_code_usage.sum()
            train_metrics["ppl"] = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10))).item()

            n_dead_train = int((train_code_usage == 0).sum().item())
            n_dead_val = int((val_code_usage == 0).sum().item())

            if epoch <= self.train_cfg["pca_wandb_epochs"]:
                self.logger.log_pca(f"Epoch {epoch}", self.model.codebook.weight.data, step=self._global_step, code_usage=val_code_usage)

            noise_scale = max(self.train_cfg["noise_init_scale"] * (1 - epoch / (noise_reinit_epochs + 1)), 0.0)

            self.logger.log_epoch(train_metrics, val_metrics, n_dead_train, n_dead_val, noise_scale, epoch, self.model.codebook.weight.data)
            self.logger.log_csv(epoch, train_metrics, val_metrics, n_dead_train, n_dead_val, noise_scale, self.model.codebook.weight.data)

            self._reinit_dead_codes(epoch, val_code_usage, noise_scale)
            self._step_scheduler()

            epoch_pbar.set_postfix({
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "ppl": f"{val_metrics['ppl']:.1f}",
                "dead": n_dead_val,
            })
            is_best = val_metrics["loss"] < best_val_loss - self.train_cfg["min_delta"]
            if is_best:
                suffix = "\t(best)"
            elif self.use_early_stopping:
                suffix = f"\t(patience {patience + 1}/{self.train_cfg['patience']})"
            else:
                suffix = ""
            tqdm.write(
                f"Epoch {epoch} | "
                f"train={train_metrics['loss']:.4f}  val={val_metrics['loss']:.4f}  "
                f"ppl={val_metrics['ppl']:.1f}  dead={n_dead_val}"
                + suffix
            )

            if epoch % self.train_cfg["ckpt_save_interval"] == 0:
                self._save_checkpoint(epoch, interval=True)

            if is_best:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics["loss"])
                patience = 0
            elif self.use_early_stopping:
                patience += 1
                if patience >= self.train_cfg["patience"]:
                    tqdm.write(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        suffix = f"_best{best_epoch}"
        self._rename_checkpoint(suffix)
        self.logger.rename_csv(suffix)
        self.logger.finish()

    def _train_epoch(self, epoch: int) -> tuple[dict, torch.Tensor]:
        self.model.train()
        loss_keys = Logger.LOSS_KEYS
        totals = dict.fromkeys(loss_keys, 0.0)
        n_batches = 0
        code_usage = torch.zeros(self.model.n_e)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False, dynamic_ncols=True)
        for hidden, target in pbar:
            if not self._reinit_done and self._global_step >= self._reinit_step:
                pbar.refresh()
                self._reinit_codebook()
                self._reinit_done = True

            hidden = hidden.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(hidden, target)
            z_e = out["z_e"].detach()
            loss = out["loss"] if self._reinit_done else out["loss"] - out["cbook_loss"]
            loss.backward()
            grad_norms = self._get_grad_norms()
            self.optimizer.step()
            code_usage += torch.bincount(out["min_indices"].detach().cpu(), minlength=self.model.n_e)

            self.logger.log_step(
                hidden, z_e, out, grad_norms,
                self._global_step, self.model.codebook.weight.data,
                loss=loss.item(), cbook_active=self._reinit_done,
            )

            if (self.train_cfg["pca_local"]
                    and epoch <= self.train_cfg["pca_local_epochs"]
                    and self._global_step % self.train_cfg["pca_local_interval"] == 0):
                self.logger.log_pca_snapshot(
                    codebook=self.model.codebook.weight.data,
                    step=self._global_step,
                    tag=f"epoch_{epoch}_step_{self._global_step:08d}",
                    z_e=z_e,
                    to_wandb=False,
                )

            self._global_step += 1

            for k in loss_keys:
                totals[k] += loss.item() if k == "loss" else out[k].item()
            n_batches += 1
            pbar.set_postfix({
                "loss": f"{totals['loss'] / n_batches:.4f}",
                "ppl": f"{totals['ppl'] / n_batches:.1f}",
            })

        return {k: v / n_batches for k, v in totals.items()}, code_usage

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[dict, torch.Tensor]:
        self.model.eval()
        loss_keys = Logger.LOSS_KEYS
        totals = dict.fromkeys(loss_keys, 0.0)
        n_batches = 0
        code_usage = torch.zeros(self.model.n_e)

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} val", leave=False, dynamic_ncols=True)
        for hidden, target in pbar:
            hidden = hidden.to(self.device)
            target = target.to(self.device)

            out = self.model(hidden, target)
            code_usage += torch.bincount(out["min_indices"].cpu(), minlength=self.model.n_e)

            for k in loss_keys:
                totals[k] += out[k].item()
            n_batches += 1
            pbar.set_postfix({
                "loss": f"{totals['loss'] / n_batches:.4f}",
                "ppl": f"{totals['ppl'] / n_batches:.1f}",
            })

        metrics = {k: v / n_batches for k, v in totals.items()}
        e_mean = code_usage / code_usage.sum()
        metrics["ppl"] = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10))).item()
        return metrics, code_usage

    def _get_grad_norms(self) -> dict:
        def norm(params):
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                return 0.0
            return torch.stack([g.norm() ** 2 for g in grads]).sum().sqrt().item()

        return {
            "encoder": norm(self.model.encoder.parameters()),
            "codebook": norm([self.model.codebook.weight]),
            "decoder": norm(self.model.decoder.parameters()),
            "classifier": norm(self.model.classifier.parameters()),
        }

    def _reset_codebook_adam_state(self):
        """Resets Adam moments for the codebook parameter to remove stale momentum after reinit."""
        state = self.optimizer.state.get(self.model.codebook.weight)
        if state:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()

    def _build_scheduler(self):
        if self.train_cfg["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_cfg["n_epochs"])
        return None

    def _step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

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
