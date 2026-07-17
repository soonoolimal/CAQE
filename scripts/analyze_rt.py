import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import yaml
from scipy.stats import chi2 as _chi2

from experiments.natural_stories_parser import load_rts, parse_natural_stories
from experiments.rt_analyzer import RTAnalyzer
from model import VQCAE

_ROOT = Path(__file__).parents[1]
_DATA_CFG = _ROOT / "configs" / "data.yaml"
_CANDIDATES = ["X_punct", "X_func", "X_eos", "X_sos"]  # forward-selection order (paper Table 5)


# per-(subject, sentence) table with response y and sentence-level covariates (entropy attached separately)
def _build_rt_table(sentences: list, rt_df: pd.DataFrame) -> pd.DataFrame:
    zone_to_sent = pd.DataFrame(
        [(s.story_id, w.zone, s.sent_id) for s in sentences for w in s.words],
        columns=["item", "zone", "sent"],
    )
    covariates = pd.DataFrame(
        [(s.story_id, s.sent_id, s.x_punct(), s.x_func(), s.x_sos(), s.x_eos()) for s in sentences],
        columns=["item", "sent", "X_punct", "X_func", "X_sos", "X_eos"],
    )

    rt = rt_df.merge(zone_to_sent, on=["item", "zone"], how="inner")
    # y = log(mean per-word RT over the subject's observed words in the sentence)
    agg = rt.groupby(["subject_id", "item", "sent"])["rt"].agg(rt_sum="sum", n_obs="count").reset_index()
    agg["y"] = np.log(agg["rt_sum"] / agg["n_obs"])

    return agg.merge(covariates, on=["item", "sent"], how="inner")


# attach per-sentence H_bar (aligned with cache order); inner join restricts rows to cached sentences
def _attach_entropy(table: pd.DataFrame, cache: dict, h_bar: list[float]) -> pd.DataFrame:
    entropy = pd.DataFrame({"item": cache["story_ids"], "sent": cache["sent_ids"], "H_bar": h_bar})
    return table.merge(entropy, on=["item", "sent"], how="inner")


# subject random intercept + story fixed effect
def _fit(df: pd.DataFrame, formula: str):
    return smf.mixedlm(formula, df, groups=df["subject_id"]).fit(reml=False, method="lbfgs")


def _ic(res, n: int) -> tuple[float, float]:
    k = res.params.shape[0] + 1  # fixed effects + group variance
    return -2 * res.llf + 2 * k, -2 * res.llf + k * np.log(n)


# forward-select covariates by LRT (p<.05) + lowest BIC. add_terms(x) gives the terms one covariate contributes;
# dof = df of that step (1 for a main effect, 2 for main + interaction). returns (selected, Table-5 trace).
def _forward_select(df: pd.DataFrame, base_terms: list[str], add_terms, dof: int) -> tuple[list[str], list]:
    n = len(df)
    thresh = _chi2.ppf(0.95, dof)  # LRT threshold at p<.05
    selected, terms = [], list(base_terms)
    current = _fit(df, "y ~ " + " + ".join(terms))
    trace = [([], *_ic(current, n), None, None)]  # baseline step: (covariates, aic, bic, chi2, p)
    remaining = list(_CANDIDATES)
    while remaining:
        best = None  # (covariate, fitted, bic, lr)
        for x in remaining:
            res = _fit(df, "y ~ " + " + ".join(terms + add_terms(x)))
            lr = 2 * (res.llf - current.llf)
            _, bic = _ic(res, n)
            if lr > thresh and (best is None or bic < best[2]):
                best = (x, res, bic, lr)
        if best is None:
            break
        x, current, _, lr = best
        selected.append(x)
        remaining.remove(x)
        terms += add_terms(x)
        trace.append((list(selected), *_ic(current, n), lr, float(_chi2.sf(lr, dof))))
    return selected, trace


# Setting A: select on H_freq (constant entropy -> interactions collapse to covariate main effects), 1 df each
def _select_A(design: pd.DataFrame) -> tuple[list[str], list]:
    return _forward_select(design, ["C(item)"], lambda x: [x], dof=1)


# Setting B: select on a varying-entropy metric; hierarchy adds main effect + interaction per covariate, 2 df each
def _select_B(metric_df: pd.DataFrame) -> tuple[list[str], list]:
    return _forward_select(metric_df, ["H_bar", "C(item)"], lambda x: [x, f"H_bar:{x}"], dof=2)


# H_freq: covariate main effects only (constant entropy absorbed); other metrics: entropy x covariate interactions
def _formula_A(selected: list[str], entropy: bool) -> str:
    terms = (["H_bar"] + [f"H_bar:{x}" for x in selected]) if entropy else list(selected)
    return "y ~ " + " + ".join(terms + ["C(item)"])


# Setting B model: entropy main effect + selected covariates as main effect + interaction (hierarchy)
def _formula_B(selected: list[str]) -> str:
    terms = ["H_bar"] + [t for x in selected for t in (x, f"H_bar:{x}")]
    return "y ~ " + " + ".join(terms + ["C(item)"])


def analyze_rt(args):
    with open(_DATA_CFG) as f:
        cfg = yaml.safe_load(f)
    backbone_cfg = cfg["backbone"][args.backbone_key]

    if args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentences = parse_natural_stories()
    print(f"Parsed {len(sentences)} unique sentences")

    analyzer = RTAnalyzer(args.backbone_key, backbone_cfg, device)
    cache = analyzer.prepare(sentences, cache_path=analyzer.h_dir / f"{args.backbone_key}.pt")

    # restrict to sentences present in the cache so all metrics compare on identical rows
    design = _attach_entropy(_build_rt_table(sentences, load_rts()), cache, RTAnalyzer.compute_mean_llm_entropy(cache))
    design = design.drop(columns="H_bar")  # shared covariate design (entropy attached per metric below)
    n = len(design)
    print(f"RT table: {n} (subject, sentence) rows, {design['subject_id'].nunique()} subjects")

    # CAQE entropy per checkpoint (H_freq constant/absorbed; H_LLM handled as baseline below)
    caqe_runs = []  # (n_e, run_name, h_bar)
    for ckpt_path in sorted((_ROOT / "logs").glob(f"{args.backbone_key}_*/checkpoints/*_best*.pt")):
        with open(ckpt_path.parents[1] / "configs.yaml") as f:
            run_cfg = yaml.safe_load(f)
        vqcae = RTAnalyzer.load_model(ckpt_path, VQCAE(run_cfg["model"]).to(device))
        n_e = int(re.search(r"_n(\d+)_", ckpt_path.name).group(1))
        caqe_runs.append((n_e, ckpt_path.parents[1].name, analyzer.compute_mean_caqe(cache, vqcae)))
        del vqcae
        torch.cuda.empty_cache()

    llm_table = _attach_entropy(design, cache, RTAnalyzer.compute_mean_llm_entropy(cache))
    selected_A, trace_A = _select_A(design)  # covariate structure from H_freq baseline (n_e-independent)
    print(f"\n[Setting A] selected covariates: {selected_A}")

    # Table 5: Setting-A forward-selection trace on H_freq (n_e-independent -> one file per backbone/run)
    m_re, m_ep = re.search(r"_re(\d+)_", caqe_runs[0][1]), re.search(r"_ep(\d+)_", caqe_runs[0][1])
    base_tag = f"{args.backbone_key}_re{m_re.group(1)}_ep{m_ep.group(1)}"
    _save_selection(trace_A, _ROOT / "results" / "reading_time" / f"{base_tag}_selection.csv")

    # baseline rows shared by every (backbone, n_e) result: H_freq (A), H_LLM (A and B)
    res = _fit(design, _formula_A(selected_A, entropy=False))
    baseline = [("A", "H_freq", "", float("nan"), *_ic(res, n), selected_A)]
    res = _fit(llm_table, _formula_A(selected_A, entropy=True))
    baseline.append(("A", "H_LLM", "", res.fe_params["H_bar"], *_ic(res, n), selected_A))
    sel_B, _ = _select_B(llm_table)
    res = _fit(llm_table, _formula_B(sel_B))
    baseline.append(("B", "H_LLM", "", res.fe_params["H_bar"], *_ic(res, n), sel_B))

    # one CSV per (backbone, n_e): baselines + that checkpoint's H_CAQE, named by its training run
    for n_e, run_name, h_bar in caqe_runs:
        caqe_table = _attach_entropy(design, cache, h_bar)
        rows = list(baseline)
        res = _fit(caqe_table, _formula_A(selected_A, entropy=True))
        rows.append(("A", "H_CAQE", n_e, res.fe_params["H_bar"], *_ic(res, n), selected_A))
        sel_B, _ = _select_B(caqe_table)
        res = _fit(caqe_table, _formula_B(sel_B))
        rows.append(("B", "H_CAQE", n_e, res.fe_params["H_bar"], *_ic(res, n), sel_B))
        run_tag = re.sub(r"_\d+$", "", run_name)  # drop training timestamp suffix
        _report(rows, _ROOT / "results" / "reading_time" / f"{run_tag}.csv")


# Table 1: metric comparison. p_theta = BIC-based Bayesian model probability within each setting.
def _report(results: list, out_path: Path):
    print(f"\n{out_path.stem}")
    print(f"{'set':>3}  {'metric':<8}{'n_e':>6}{'beta_H':>10}{'AIC':>12}{'BIC':>12}{'dBIC':>10}{'p_theta':>9}  covariates")
    csv_rows = []
    for setting in ("A", "B"):
        rows = [r for r in results if r[0] == setting]
        best = min(r[5] for r in rows)
        weights = [np.exp(-0.5 * (r[5] - best)) for r in rows]
        total = sum(weights)
        for (_, metric, n_e, beta, aic, bic, cov), w in zip(rows, weights):
            dbic, p_theta = bic - best, w / total
            print(f"{setting:>3}  {metric:<8}{str(n_e):>6}{beta:>10.4f}{aic:>12.1f}{bic:>12.1f}{dbic:>10.1f}{p_theta:>9.3f}  {'+'.join(cov)}")
            csv_rows.append([setting, metric, n_e, round(beta, 6), round(aic, 2), round(bic, 2), round(dbic, 2), round(p_theta, 4), "+".join(cov)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "metric", "n_e", "beta_H", "aic", "bic", "delta_bic", "p_theta", "covariates"])
        writer.writerows(csv_rows)


# Table 5: forward-selection trace (each step adds one covariate; chi2/p from LRT vs previous step). best = lowest BIC.
def _save_selection(trace: list, out_path: Path):
    best_bic = min(t[2] for t in trace)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{out_path.stem} (Table 5 -- forward selection)")
    print(f"{'covariates':<38}{'AIC':>11}{'BIC':>11}{'dBIC':>9}{'chi2':>10}{'p':>10}{'best':>6}")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["covariates", "aic", "bic", "delta_bic", "chi2", "p", "best"])
        for cov, aic, bic, chi2, p in trace:
            spec = "+".join(cov) if cov else "baseline"
            chi2_s, p_s = ("", "") if chi2 is None else (f"{chi2:.2f}", f"{p:.2e}")
            star = "*" if bic == best_bic else ""
            writer.writerow([spec, round(aic, 2), round(bic, 2), round(bic - best_bic, 2), chi2_s, p_s, star])
            print(f"{spec:<38}{aic:>11.1f}{bic:>11.1f}{bic - best_bic:>9.1f}{chi2_s:>10}{p_s:>10}{star:>6}")
