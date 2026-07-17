import csv
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

from experiments import Evaluator
from experiments.coinco_parser import not_found_rate, parse_coinco
from model import VQCAE

_ROOT = Path(__file__).parents[1]
_DATA_CFG = _ROOT / "configs" / "data.yaml"


def load_configs(backbone_key: str) -> tuple[dict, Path]:
    with open(_DATA_CFG) as f:
        data_cfg = yaml.safe_load(f)
    with open(_ROOT / "configs" / "experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    backbone_cfg = data_cfg["backbone"][backbone_key]
    coinco_path = _ROOT / exp_cfg["datasets"]["coinco"]["path"]

    return backbone_cfg, coinco_path


def _parse_ckpt_name(ckpt_name: str, backbone_key: str) -> tuple[str, str, int | None, int | None]:
    if ckpt_name == backbone_key:
        return "(baseline)", "baseline", None, None

    m_ne = re.search(r"_n(\d+)_", ckpt_name)
    n_e = int(m_ne.group(1)) if m_ne else None

    for pattern, kind in ((r"_epoch(\d+)\.pt$", "epoch"), (r"_best(\d+)\.pt$", "best")):
        m = re.search(pattern, ckpt_name)
        if m:
            return ckpt_name[: m.start()], kind, int(m.group(1)), n_e

    return ckpt_name.removesuffix(".pt"), "other", None, n_e


def _epoch_label(kind: str, epoch: int | None) -> str:
    if kind == "baseline":
        return "(baseline)"
    if kind == "best":
        return f"{epoch} (best)"
    if kind == "epoch":
        return str(epoch)
    return ""


def _epoch_sort_key(epoch_label: str) -> int:
    epoch_label = str(epoch_label)
    if epoch_label == "(baseline)":
        return -1
    if epoch_label.endswith("(best)"):
        return 10**9
    return int(epoch_label) if epoch_label.isdigit() else 10**9 + 1


def _results_dir(backbone_key: str, run_name: str) -> Path:
    m_re = re.search(r"_re(\d+)_", run_name)
    m_ep = re.search(r"_ep(\d+)_", run_name)
    re_s = m_re.group(1) if m_re else "NA"
    ep_s = m_ep.group(1) if m_ep else "NA"
    return _ROOT / "results" / "eval" / f"{backbone_key}_re{re_s}_ep{ep_s}"


def _ckpt_sort_key(ckpt_path: Path, backbone_key: str) -> tuple[int, int]:
    _, kind, epoch, n_e = _parse_ckpt_name(ckpt_path.name, backbone_key)
    return (n_e if n_e is not None else -1, _epoch_sort_key(_epoch_label(kind, epoch)))


def _write_summary(eval_csv: Path, out_path: Path):
    df = pd.read_csv(eval_csv, dtype={"epoch": str})
    agg = (
        df.groupby(["n_e", "epoch"], dropna=False, sort=False)["delta_p"]
        .agg(mean="mean", std="std", count="count", mean_abs=lambda s: s.abs().mean())
        .reset_index()
    )

    rows = []
    for _, r in agg.iterrows():
        epoch_label, n_e = r["epoch"], r["n_e"]
        rows.append(
            {
                "n_e": (pd.NA if pd.isna(n_e) else int(n_e)),
                "epoch": epoch_label,
                "mean_delta_p": round(r["mean"], 6),
                "mean_abs": round(r["mean_abs"], 6),
                "std": round(r["std"], 6),
                "n_pairs": int(r["count"]),
                "_n_e_key": -1 if pd.isna(n_e) else int(n_e),
                "_epoch_key": _epoch_sort_key(epoch_label),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["_n_e_key", "_epoch_key"], kind="stable")
    summary = summary[["n_e", "epoch", "mean_delta_p", "mean_abs", "std", "n_pairs"]]
    summary["n_e"] = summary["n_e"].astype("Int64")  # nullable int keeps the baseline's n_e cell empty
    summary.to_csv(out_path, index=False)


def eval_model(args):
    backbone_cfg, coinco_path = load_configs(args.backbone_key)

    if args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = parse_coinco(coinco_path)
    print(f"CoinCo entries loaded: {len(entries)}")
    print(f"Target not found: {not_found_rate(entries):.1%} (backbone-independent skip)")

    evaluator = Evaluator(args.backbone_key, backbone_cfg, device)
    evaluator.prepare(entries, cache_path=evaluator.h_dir / f"{args.backbone_key}.pt")

    ckpt_paths = sorted(
        (_ROOT / "logs").glob(f"{args.backbone_key}_*/checkpoints/*.pt"),
        key=lambda p: _ckpt_sort_key(p, args.backbone_key),
    )
    print(f"Found {len(ckpt_paths)} checkpoints for {args.backbone_key}")

    run_name = ckpt_paths[0].parents[1].name if ckpt_paths else args.backbone_key
    out_path = _results_dir(args.backbone_key, run_name) / "eval.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_e", "epoch", "target_word", "substitute_word", "delta_p", "p_target", "p_sub"])

        # pure LLM baseline (no VQCAE, backbone logits only)
        llm_results = evaluator.evaluate_llm()
        for target_word, substitute_word, delta, p_t, p_s in llm_results:
            writer.writerow(["", "(baseline)", target_word, substitute_word, delta, p_t, p_s])
        llm_deltas = [d for _, _, d, _, _ in llm_results]
        print(f"{args.backbone_key} (llm): mean={sum(llm_deltas) / len(llm_deltas):.6f}  n_pairs={len(llm_deltas)}")
        evaluator.unload()

        # VQCAE checkpoints
        for ckpt_path in ckpt_paths:
            with open(ckpt_path.parents[1] / "configs.yaml") as cfg_f:
                run_cfg = yaml.safe_load(cfg_f)

            vqcae = Evaluator.load_model(ckpt_path, VQCAE(run_cfg["model"]).to(device))
            results = evaluator.evaluate(vqcae)

            _, kind, epoch, n_e = _parse_ckpt_name(ckpt_path.name, args.backbone_key)
            epoch_cell, n_e_cell = _epoch_label(kind, epoch), ("" if n_e is None else n_e)

            for target_word, substitute_word, delta, p_t, p_s in results:
                writer.writerow([n_e_cell, epoch_cell, target_word, substitute_word, delta, p_t, p_s])

            deltas = [d for _, _, d, _, _ in results]
            print(f"{ckpt_path.name}: mean={sum(deltas) / len(deltas):.6f}  n_pairs={len(deltas)}")

            del vqcae
            torch.cuda.empty_cache()

    summary_path = out_path.parent / "eval_summary.csv"
    _write_summary(out_path, summary_path)

    print(f"Results saved to {out_path}")
    print(f"Summary saved to {summary_path}")
