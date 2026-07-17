import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
}


def plot_quant_error(run_dir: Path):
    run_name = run_dir.name
    diag_csv = run_dir / f"{run_name}_diagnose.csv"

    if not diag_csv.exists():
        print(f"No diagnostic .csv at {diag_csv}")
        return

    steps, qe = [], []
    with open(diag_csv) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            qe.append(float(row["quant_error"]))

    reinit_steps = None
    configs_path = run_dir / "configs.yaml"
    if configs_path.exists():
        with open(configs_path) as f:
            cfg = yaml.safe_load(f)
        reinit_steps = cfg.get("train", {}).get("reinit_steps")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, qe, linewidth=0.8, color="#4C9BE8", alpha=0.7, label="quant_error")

        if reinit_steps is not None and reinit_steps < len(qe):
            qe_at_reinit = qe[reinit_steps]
            ax.plot(
                [reinit_steps, reinit_steps],
                [0, qe_at_reinit],
                color="#FF4500",
                linestyle="--",
                linewidth=1.2,
                label=f"reinit_steps={reinit_steps}",
            )
            ax.scatter([reinit_steps], [qe_at_reinit], color="#FF4500", s=40, zorder=5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("step")
        ax.set_ylabel("quant_error")
        ax.set_title(f"Diagnostic: {run_name}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)

        fig.tight_layout()

    out = run_dir / f"{run_name}_diagnose.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
