from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

_ROOT = Path(__file__).parents[1]

_DISPLAY_NAMES = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "modernbert": "ModernBERT",
    "opt_1_3b": "OPT-1.3B",
    "llama31_8b": "LLaMA-3.1-8B",
    "llama32_3b": "LLaMA-3.2-3B",
}


def plot_eval_violin(args):
    matches = sorted((_ROOT / "results" / "eval").glob(f"{args.backbone_key}_re*_ep*"))
    if not matches:
        raise FileNotFoundError(f"no results dir for {args.backbone_key} under {_ROOT / 'results'}")
    out_dir = matches[-1]
    df = pd.read_csv(out_dir / "eval.csv", dtype={"epoch": str})

    display_name = _DISPLAY_NAMES.get(args.backbone_key, args.backbone_key)

    llm_deltas = df[df["epoch"] == "(baseline)"]["delta_p"].tolist()
    n_e_values = sorted(int(v) for v in df["n_e"].dropna().unique())

    for n_e in n_e_values:
        sub = df[df["n_e"] == n_e]
        epoch_labels = list(sub["epoch"].unique())

        best_map = {int(lbl.split()[0]): lbl for lbl in epoch_labels if str(lbl).endswith("(best)")}
        groups = [(display_name, llm_deltas)]
        for ep_num, label in sorted((int(e), e) for e in epoch_labels if str(e).isdigit()):
            groups.append((label, sub[sub["epoch"] == label]["delta_p"].tolist()))
            if ep_num in best_map:
                best_label = best_map[ep_num]
                groups.append((best_label, sub[sub["epoch"] == best_label]["delta_p"].tolist()))

        labels = [g[0] for g in groups]
        data = [g[1] for g in groups]

        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 0.8), 5), layout="constrained")
        parts: Any = ax.violinplot(data, positions=range(len(data)), showmedians=False, showextrema=False)

        for i, (label, _) in enumerate(groups):
            if label == display_name:
                parts["bodies"][i].set_facecolor("#e07b7b")
            elif str(label).endswith("(best)"):
                parts["bodies"][i].set_facecolor("#2563eb")
            else:
                parts["bodies"][i].set_facecolor("#2ca8a0")
            parts["bodies"][i].set_alpha(0.85)

        for i, d in enumerate(data):
            arr = np.array(d)
            mean = arr.mean()
            std = arr.std()
            ax.vlines(i, mean - std, mean + std, colors="#111111", linewidths=0.8)
            ax.hlines([mean - std, mean + std], i - 0.15, i + 0.15, colors="#111111", linewidths=0.8)
            ax.plot(i, mean, "o", color="#cc3333", markersize=4, zorder=3)

        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray", zorder=0)  # P(target) == P(substitute)
        ax.set_ylim(-1, 1)  # signed delta_p is bounded to [-1, 1]
        ax.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.3)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\Delta \mathrm{P} = \mathrm{P}(\mathrm{target}) - \mathrm{P}(\mathrm{substitute})$")

        st = fig.suptitle(
            r"Distribution of $\Delta \mathrm{P}$" + "\n" + rf"{display_name} ($n_e = {n_e}$)",
            fontsize=16,
            fontweight="normal",
        )

        fig.canvas.draw()
        fig.set_layout_engine("none")
        st.set_x(ax.get_position().x0 + ax.get_position().width / 2)
        out_path = out_dir / f"plot_violin_n{n_e}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def plot_eval_cross(args):
    matches = sorted((_ROOT / "results" / "eval").glob(f"{args.backbone_key}_re*_ep*"))
    if not matches:
        raise FileNotFoundError(f"no results dir for {args.backbone_key} under {_ROOT / 'results'}")
    out_dir = matches[-1]
    df = pd.read_csv(out_dir / "eval.csv", dtype={"epoch": str})

    display_name = _DISPLAY_NAMES.get(args.backbone_key, args.backbone_key)
    baseline_rows = df[df["epoch"] == "(baseline)"]
    n_e_values = sorted(int(v) for v in df["n_e"].dropna().unique())

    _CHUNK = 5  # epochs per subplot

    for n_e in n_e_values:
        sub = df[df["n_e"] == n_e]
        epoch_labels = list(sub["epoch"].unique())

        numeric_epochs = sorted((int(e), e) for e in epoch_labels if str(e).isdigit())
        best_labels = [label for label in epoch_labels if str(label).endswith("(best)")]

        chunks = [numeric_epochs[i : i + _CHUNK] for i in range(0, len(numeric_epochs), _CHUNK)]

        # global teal gradient across all epochs for consistent color progression
        teal_cmap = mcolors.LinearSegmentedColormap.from_list("teal", ["#d0f0ec", "#0a4a45"])
        n_ep = len(numeric_epochs)
        ep_color = {label: teal_cmap(i / max(n_ep - 1, 1)) for i, (_, label) in enumerate(numeric_epochs)}

        # baseline appears in every subplot, best is placed only in its own chunk
        fixed: list[tuple[str, Any]] = [("(baseline)", baseline_rows)]
        best_epoch_map = {int(label.split()[0]): label for label in best_labels}

        n_cols = len(chunks)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 5.0), layout="constrained")
        if n_cols == 1:
            axes = [axes]

        # assign each best epoch to the chunk whose epochs are closest
        best_chunk: dict[int, str] = {}
        for ep_num, best_label in best_epoch_map.items():
            target = min(range(len(chunks)), key=lambda i: min(abs(ep_num - n) for n, _ in chunks[i]))
            best_chunk[target] = best_label

        for chunk_idx, (ax, chunk) in enumerate(zip(axes, chunks)):
            groups: list[tuple[str, Any]] = list(fixed)
            for _, label in chunk:
                groups.append((label, sub[sub["epoch"] == label]))
            if chunk_idx in best_chunk:
                best_label = best_chunk[chunk_idx]
                groups.append((best_label, sub[sub["epoch"] == best_label]))

            all_vals = np.concatenate([rows[["p_target", "p_sub"]].values.ravel() for _, rows in groups])
            lo, hi = all_vals.min(), all_vals.max()
            pad = (hi - lo) * 0.15
            lim = (lo - pad, hi + pad)

            ax.plot(lim, lim, color="#2c5f8a", linewidth=0.8, zorder=0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")

            handles = []
            for label, rows in groups:
                pt = rows["p_target"].values
                ps = rows["p_sub"].values
                mx, sx = pt.mean(), pt.std()
                my, sy = ps.mean(), ps.std()

                if label == "(baseline)":
                    color = "#cc3333"
                    legend_label = display_name
                    zorder = 3
                elif str(label).endswith("(best)"):
                    color = "#2563eb"
                    legend_label = f"Epoch {label.split()[0]} (best)"
                    zorder = 3
                else:
                    color = ep_color[label]
                    legend_label = f"Epoch {label}"
                    zorder = 2

                ax.plot([mx - sx, mx + sx], [my, my], color=color, linewidth=1.5, solid_capstyle="butt", zorder=zorder)
                ax.plot([mx, mx], [my - sy, my + sy], color=color, linewidth=1.5, solid_capstyle="butt", zorder=zorder)
                h = ax.plot(mx, my, "o", color=color, markersize=4, zorder=zorder + 1, label=legend_label)[0]
                handles.append(h)

            ep_start, ep_end = chunk[0][1], chunk[-1][1]
            ax.set_xlabel(r"$\mathrm{P}(\mathrm{target})$", fontsize=9)
            ax.set_ylabel(r"$\mathrm{P}(\mathrm{substitute})$", fontsize=9)
            ax.set_title(f"Epoch {ep_start}–{ep_end}", fontsize=10, fontweight="normal")
            ax.grid(linestyle="-", linewidth=0.4, alpha=0.3)
            ax.legend(handles=handles, fontsize=7, loc="best")

        fig.suptitle(
            rf"$\mathrm{{P}}(\mathrm{{target}})$ vs. $\mathrm{{P}}(\mathrm{{substitute}})$"
            "\n"
            rf"{display_name} ($n_e = {n_e}$)",
            fontsize=16,
            fontweight="normal",
        )

        out_path = out_dir / f"plot_cross_n{n_e}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")
