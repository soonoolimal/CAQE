import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from matplotlib.figure import Figure
from PIL import Image as PILImage
from sklearn.decomposition import PCA


class Logger:
    LOSS_KEYS = [
        "loss",
        "cbook_loss",
        "commit_loss",
        "recon_loss",
        "clf_loss",
        "cbook_ppl",
        "clf_ppl",
        "clf_acc",
    ]  # + n_dead

    _RC = {
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }

    _SCATTER_STYLE = {
        "cbook_init": dict(s=6, alpha=0.6, color="#222222", zorder=1),
        "cbook_active": dict(
            s=50, alpha=0.9, color="#2CA02C", marker="*", edgecolors="black", linewidths=0.3, zorder=4
        ),
        "cbook_dead": dict(s=10, alpha=0.7, color="#FF4500", marker="x", zorder=3),
        "z_e": dict(s=3, alpha=0.35, color="#1A6FBF", zorder=2),
    }

    def __init__(
        self,
        run_name: str,
        train_cfg: dict,
        model_cfg: dict,
        wandb_key: str | None = None,
        wandb_entity: str | None = None,
    ):
        if wandb_key:
            wandb.login(key=wandb_key, relogin=True)

        wandb.init(
            project=train_cfg["wandb_project"],
            entity=wandb_entity,
            name=run_name,
            config={"train": train_cfg, "model": model_cfg},
        )
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("valid/*", step_metric="epoch")

        self.pca_wandb: bool = train_cfg["pca_wandb"]

        run_dir = Path(train_cfg["local_log_dir"]) / run_name
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True)

        # save configs combined for run reproducibility
        with open(run_dir / "configs.yaml", "w") as f:
            yaml.dump({"train": train_cfg, "model": model_cfg}, f, default_flow_style=False)

        self._csv_path, self._csv_fields = self._init_epoch_csv(run_dir, run_name)
        self._diag_csv_path = self._init_diag_csv(run_dir, run_name) if train_cfg["diagnose"] else None

        self._pca: PCA | None = None
        self._cbook_init_2d: np.ndarray | None = None

    def fit_pca(self, codebook: torch.Tensor):
        """Fit PCA axes from the reinit 1 codebook. All subsequent snapshots reuse these fixed axes."""
        cbook = codebook.detach().cpu().numpy()
        self._pca = PCA(n_components=2, random_state=0)
        self._pca.fit(cbook)
        self._cbook_init_2d = self._pca.transform(cbook)

    def log_pca(
        self, tag: str | int, z_e: torch.Tensor, codebook: torch.Tensor, active_mask: torch.Tensor | None = None
    ):
        if self._pca is None:
            return

        cbook_2d = self._pca.transform(codebook.detach().cpu().numpy())
        ze_2d = self._pca.transform(z_e.detach().cpu().numpy())
        active_np = active_mask.numpy() if active_mask is not None else None

        fig = self._make_pca_fig(tag, ze_2d, cbook_2d, active_np)

        key = f"step_{tag:04d}" if isinstance(tag, int) else tag
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        wandb.log({f"pca/{key}": wandb.Image(PILImage.open(buf))})

        plt.close(fig)

    def log_step(self, step: int, quant_error: float):
        wandb.log({"diagnostic/quant_error": quant_error})

        if self._diag_csv_path is not None:
            with open(self._diag_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([step, quant_error])

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        wandb.log(
            {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"valid/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
            }
        )

        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(
                {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"valid_{k}": v for k, v in val_metrics.items()},
                }
            )

    def _make_pca_fig(
        self, tag: str | int, ze_2d: np.ndarray, cbook_2d: np.ndarray, active_np: np.ndarray | None = None
    ) -> Figure:
        with plt.rc_context(self._RC):
            fig, ax = plt.subplots(figsize=(8, 6))

            # encoder output distribution
            ax.scatter(ze_2d[:, 0], ze_2d[:, 1], **self._SCATTER_STYLE["z_e"], label=f"z_e ({len(ze_2d):,})")

            # Reinit 1 codebook positions as fixed reference overlay
            if tag != "reinit1":
                ax.scatter(
                    self._cbook_init_2d[:, 0],
                    self._cbook_init_2d[:, 1],
                    **self._SCATTER_STYLE["cbook_init"],
                    label="codebook (Reinit 1)",
                )

            # current codebook positions at snapshot time, split by active/dead
            if active_np is not None:
                dead_2d = cbook_2d[~active_np]
                active_2d = cbook_2d[active_np]
                # dead codes (not assigned by any z_e in this snapshot)
                if len(dead_2d) > 0:
                    ax.scatter(
                        dead_2d[:, 0],
                        dead_2d[:, 1],
                        **self._SCATTER_STYLE["cbook_dead"],
                        label=f"dead ({(~active_np).sum():,})",
                    )
                # active codes
                ax.scatter(
                    active_2d[:, 0],
                    active_2d[:, 1],
                    **self._SCATTER_STYLE["cbook_active"],
                    label=f"active ({active_np.sum():,})",
                )
            # fallback: active_mask not provided
            else:
                ax.scatter(
                    cbook_2d[:, 0],
                    cbook_2d[:, 1],
                    **self._SCATTER_STYLE["cbook_active"],
                    label=f"codebook ({len(cbook_2d):,})",
                )

            if tag == "reinit1":
                _title = "After K-Means Init (Reinit 1)"
            elif tag == "reinit2":
                _title = "After K-Means Reinit (Reinit 2)"
            else:
                _title = f"Pre-Reinit (step {tag})"

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(_title)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3, markerscale=1.5)

            fig.tight_layout()

        return fig

    def _init_epoch_csv(self, run_dir: Path, run_name: str) -> tuple[Path, list[str]]:
        metric_keys = self.LOSS_KEYS + ["n_dead"]
        fields = ["epoch"] + [f"train_{k}" for k in metric_keys] + [f"valid_{k}" for k in metric_keys]
        path = run_dir / f"{run_name}.csv"
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        return path, fields

    def _init_diag_csv(self, run_dir: Path, run_name: str) -> Path:
        path = run_dir / f"{run_name}_diagnose.csv"
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "quant_error"])
        return path

    def finish(self):
        wandb.finish()
