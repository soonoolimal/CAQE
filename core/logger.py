import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import wandb


class Logger:
    LOSS_KEYS = ["loss", "cbook_loss", "commit_loss", "recon_loss", "clf_loss", "clf_acc", "ppl"]
    PCA_ZE_SAMPLES = 5000
    _SCATTER_STYLE = {
        "init": dict(s=2, alpha=0.1, color="#CCCCCC", zorder=1),
        "dead": dict(s=8, alpha=0.7, color="#000000", zorder=3, marker="x"),
        "active": dict(s=4, alpha=0.5, color="#FF4500", zorder=2),
        "s1": dict(s=3, alpha=0.3, color="#2ECC71", zorder=2),
        "s2": dict(s=3, alpha=0.5, color="#FF4500", zorder=3),
        "z_e": dict(s=4, alpha=0.2, color="#4C9BE8", zorder=0),
    }

    def __init__(self, project: str, run_name: str, config: dict):
        wandb.init(project=project, name=run_name, config=config)
        wandb.define_metric("global_step")
        wandb.define_metric("step/*", step_metric="global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("codebook/*", step_metric="epoch")
        wandb.define_metric("diagnostic/step")
        wandb.define_metric("diagnostic/*", step_metric="diagnostic/step")

        self._pca_local: bool = config["pca_local"]
        self._pca_wandb: bool = config["pca_wandb"]

        run_dir = Path(config["local_log_dir"]) / run_name
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.pca_snap_dir = run_dir / "pca"
        if self._pca_local:
            self.pca_snap_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path, self._csv_fields, self._flat_cfg = self._init_csv(run_dir, run_name, config)
        self._diagnostic_csv_path = run_dir / "diagnostic_quant_error.csv"
        with open(self._diagnostic_csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["step", "quant_error"]).writeheader()

        self._pca: PCA | None = None
        self._cb_init_2d: np.ndarray | None = None

    def update_config(self, updates: dict):
        wandb.config.update(updates, allow_val_change=True)

    def log_codebook_init(self, cbook: torch.Tensor):
        wandb.summary.update({
            "codebook/init_mean": cbook.mean().item(),
            "codebook/init_std": cbook.std().item(),
            "codebook/init_min": cbook.min().item(),
            "codebook/init_max": cbook.max().item(),
        })

    def log_codebook_reinit(self, cbook: torch.Tensor):
        wandb.summary.update({
            "codebook/reinit_mean": cbook.mean().item(),
            "codebook/reinit_std": cbook.std().item(),
            "codebook/reinit_min": cbook.min().item(),
            "codebook/reinit_max": cbook.max().item(),
        })

    def log_step(
        self,
        hidden: torch.Tensor,
        z_e: torch.Tensor,
        out: dict,
        grad_norms: dict,
        step: int,
        codebook: torch.Tensor,
        loss: float,
        cbook_active: bool,
    ):
        cb = codebook.detach()
        wandb.log({
            "global_step": step,
            "step/loss": loss,
            "step/cbook_active": int(cbook_active),
            "step/cbook_loss": out["cbook_loss"].item(),
            "step/commit_loss": out["commit_loss"].item(),
            "step/recon_loss": out["recon_loss"].item(),
            "step/clf_loss": out["clf_loss"].item(),
            "step/clf_acc": out["clf_acc"].item(),
            "step/quant_error": out["quant_error"].item(),
            "step/ppl": out["ppl"].item(),
            "step/hidden_mean": hidden.mean().item(),
            "step/hidden_std": hidden.std().item(),
            "step/z_e_mean": z_e.mean().item(),
            "step/z_e_std": z_e.std().item(),
            "step/grad_encoder": grad_norms["encoder"],
            "step/grad_codebook": grad_norms["codebook"],
            "step/grad_decoder": grad_norms["decoder"],
            "step/grad_classifier": grad_norms["classifier"],
            "step/codebook_mean": cb.mean().item(),
            "step/codebook_std": cb.std().item(),
            "step/codebook_min": cb.min().item(),
            "step/codebook_max": cb.max().item(),
        })

    def log_epoch(
        self,
        train_metrics: dict,
        val_metrics: dict,
        n_dead_train: int,
        n_dead_val: int,
        noise_scale: float,
        epoch: int,
        codebook: torch.Tensor,
    ):
        cb = codebook.detach()
        wandb.log({
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "train/dead_codes": n_dead_train,
            "val/dead_codes": n_dead_val,
            "train/noise_scale": noise_scale,
            "codebook/mean": cb.mean().item(),
            "codebook/std": cb.std().item(),
            "codebook/min": cb.min().item(),
            "codebook/max": cb.max().item(),
            "epoch": epoch,
        })

    def log_diagnostic_step(self, quant_error: float, step: int):
        wandb.log({"diagnostic/quant_error": quant_error, "diagnostic/step": step})
        with open(self._diagnostic_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["step", "quant_error"]).writerow({"step": step, "quant_error": quant_error})

    def fit_pca(self, codebook: torch.Tensor):
        data = codebook.detach().cpu().numpy()
        self._pca = PCA(n_components=2, random_state=0)
        self._pca.fit(data)
        self._cb_init_2d = self._pca.transform(data)

    def log_pca(self, tag: str, codebook: torch.Tensor, step: int, **kwargs):
        self.log_pca_snapshot(
            codebook=codebook,
            step=step,
            tag=tag,
            to_wandb=self._pca_wandb,
            **kwargs,
        )

    def log_pca_snapshot(
        self,
        codebook: torch.Tensor,
        step: int,
        tag: str,
        z_e: torch.Tensor | None = None,
        code_usage: torch.Tensor | None = None,
        subtitle: str | None = None,
        to_wandb: bool = True,
        local_path: Path | None = None,
        show_init: bool = False,
        s1_codebook: torch.Tensor | None = None,
        s2_codebook: torch.Tensor | None = None,
    ):
        if self._pca is None:
            return

        if local_path is None and self._pca_local:
            local_path = self.pca_snap_dir / f"{tag}.png"

        cb_2d = self._pca.transform(codebook.detach().cpu().numpy())

        _rc = {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
        with plt.rc_context(_rc):
            fig, ax = plt.subplots(figsize=(8, 6))

            if code_usage is not None:
                dead_mask = (code_usage == 0).numpy()
                alive_mask = ~dead_mask
                n_dead = int(dead_mask.sum())
                if dead_mask.any():
                    ax.scatter(cb_2d[dead_mask, 0], cb_2d[dead_mask, 1], **self._SCATTER_STYLE["dead"], label=f"dead ({n_dead:,})")
                ax.scatter(cb_2d[alive_mask, 0], cb_2d[alive_mask, 1], **self._SCATTER_STYLE["active"], label=f"active ({alive_mask.sum():,})")
            else:
                if show_init:
                    ax.scatter(self._cb_init_2d[:, 0], self._cb_init_2d[:, 1], **self._SCATTER_STYLE["init"], label="init codebook")
                if s1_codebook is not None:
                    s1_2d = self._pca.transform(s1_codebook.detach().cpu().numpy())
                    ax.scatter(s1_2d[:, 0], s1_2d[:, 1], **self._SCATTER_STYLE["s1"], label="codebook after stage 1")
                if s2_codebook is not None:
                    s2_2d = self._pca.transform(s2_codebook.detach().cpu().numpy())
                    ax.scatter(s2_2d[:, 0], s2_2d[:, 1], **self._SCATTER_STYLE["s2"], label="codebook after stage 2")
                if not show_init and s1_codebook is None and s2_codebook is None:
                    ax.scatter(cb_2d[:, 0], cb_2d[:, 1], **self._SCATTER_STYLE["s1"], label="codebook")

            if z_e is not None:
                ze_np = z_e.detach().cpu().numpy()
                ze_2d = self._pca.transform(ze_np)
                ax.scatter(ze_2d[:, 0], ze_2d[:, 1], **self._SCATTER_STYLE["z_e"], label=f"z_e ({ze_np.shape[0]:,})")

            title = f"{tag}  (step {step})"
            if subtitle:
                title += f"\n{subtitle}"
            ax.set_title(title)
            ax.legend(loc="upper right", markerscale=1.5)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            fig.tight_layout()

            if local_path is not None:
                fig.savefig(local_path, dpi=120, bbox_inches="tight")
            if to_wandb:
                wandb.log({f"pca/{tag}": wandb.Image(fig)})

            plt.close(fig)

    def _init_csv(self, run_dir: Path, run_name: str, config: dict) -> tuple[Path, list[str], dict]:
        flat_cfg = {}
        for k, v in config.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat_cfg[f"{k}.{k2}"] = v2
            else:
                flat_cfg[k] = v

        fields = (
            ["epoch"]
            + [f"train_{k}" for k in self.LOSS_KEYS]
            + [f"val_{k}" for k in self.LOSS_KEYS]
            + ["train_dead_codes", "dead_codes", "noise_scale"]
            + ["codebook_mean", "codebook_std", "codebook_min", "codebook_max"]
            + list(flat_cfg.keys())
        )

        path = run_dir / f"{run_name}.csv"
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

        return path, fields, flat_cfg

    def log_csv(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        n_dead_train: int,
        n_dead_val: int,
        noise_scale: float,
        codebook: torch.Tensor,
    ):
        cb = codebook.detach()
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "train_dead_codes": n_dead_train,
                "dead_codes": n_dead_val,
                "noise_scale": noise_scale,
                "codebook_mean": cb.mean().item(),
                "codebook_std": cb.std().item(),
                "codebook_min": cb.min().item(),
                "codebook_max": cb.max().item(),
                **self._flat_cfg,
            })

    def rename_csv(self, suffix: str):
        if self._csv_path.exists():
            self._csv_path.rename(self._csv_path.with_stem(self._csv_path.stem + suffix))

    def finish(self):
        wandb.finish()
