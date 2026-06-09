"""
Delta-P experiment on CoInCo: ΔP = P(substitute|context) - P(original|context) per method.

Usage (from project root):
    python scripts/delta_p.py --backbone mlm --model bert --ckpt path/to/ckpt.pt --coinco data/inference/coinco_dataset.txt
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from models.caqe import CAQE


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_coinco(path: str) -> list[dict]:
    """Parses CoInCo txt file into a list of entries.

    Each entry: {target, pos, sentence, substitutes: {word: count}}
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" | ")
            if len(parts) != 3:
                continue

            target_pos, sentence, subs_str = parts
            target_word, pos = target_pos.rsplit(".", 1)

            substitutes = {}
            for token in subs_str.split():
                word, count = token.rsplit("::", 1)
                substitutes[word] = int(count)

            entries.append({
                "target": target_word,
                "pos": pos,
                "sentence": sentence,
                "substitutes": substitutes,
            })
    return entries


# ---------------------------------------------------------------------------
# Token position alignment
# ---------------------------------------------------------------------------

def find_token_pos(tokenizer, sentence: str, word: str) -> int | None:
    """Returns the first token index of word in the tokenized sentence.

    Uses offset mapping from a fast tokenizer.
    """
    lower = sentence.lower()
    char_start = lower.find(word.lower())
    if char_start == -1:
        return None

    encoded = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    for i, (start, end) in enumerate(encoded["offset_mapping"]):
        if start <= char_start < end:
            return i
    return None


# ---------------------------------------------------------------------------
# Per-method probability distributions
# ---------------------------------------------------------------------------

def mlm_probs(model, tokenizer, sentence: str, token_pos: int, device) -> torch.Tensor:
    """Returns softmax distribution [vocab_size] at the masked token position."""
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    masked_ids = input_ids.clone()
    masked_ids[0, token_pos] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits
    return F.softmax(logits[0, token_pos], dim=-1).cpu()


def ntp_probs(model, tokenizer, sentence: str, token_pos: int, device) -> torch.Tensor:
    """Returns softmax distribution [vocab_size] at position token_pos - 1."""
    if token_pos == 0:
        return None
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return F.softmax(logits[0, token_pos - 1], dim=-1).cpu()


def caqe_probs(caqe_model, backbone, tokenizer, sentence: str, token_pos: int,
               backbone_type: str, device) -> torch.Tensor:
    """Returns CAQE classifier softmax distribution [vocab_size] at the target position."""
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        if backbone_type == "mlm":
            masked_ids = input_ids.clone()
            masked_ids[0, token_pos] = tokenizer.mask_token_id
            h = backbone(input_ids=masked_ids, attention_mask=attention_mask).last_hidden_state
            h = h[0, token_pos].unsqueeze(0)         # [1, hidden_dim]
        else:
            if token_pos == 0:
                return None
            h = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h = h[0, token_pos - 1].unsqueeze(0)     # [1, hidden_dim]

        z_q = caqe_model.encode_and_quantize(h)
        h_hat = caqe_model.vqvae.decoder(z_q)
        logits = caqe_model.classifier(h_hat)

    return F.softmax(logits[0], dim=-1).cpu()


def build_freq_probs(tokenizer, entries: list[dict], vocab_size: int) -> torch.Tensor:
    """Builds a unigram frequency distribution over all CoInCo sentences."""
    counts = Counter()
    special_ids = set(tokenizer.all_special_ids)
    for entry in entries:
        ids = tokenizer(entry["sentence"], add_special_tokens=False)["input_ids"]
        counts.update(tid for tid in ids if tid not in special_ids)

    freq = torch.zeros(vocab_size)
    for tid, cnt in counts.items():
        freq[tid] = cnt
    total = freq.sum()
    return freq / total if total > 0 else freq


# ---------------------------------------------------------------------------
# ΔP computation
# ---------------------------------------------------------------------------

def word_prob(probs: torch.Tensor, tokenizer, word: str) -> float | None:
    """Returns P(first subword token of word) from the distribution."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    return probs[ids[0]].item()


def compute_delta_p(probs: torch.Tensor, tokenizer, target: str,
                    substitutes: dict[str, int]) -> float | None:
    """Returns ΔP = P(best_substitute|context) - P(original|context).

    Best substitute is the one with the highest annotation count.
    """
    p_target = word_prob(probs, tokenizer, target)
    if p_target is None:
        return None

    best_sub = max(substitutes, key=substitutes.get)
    p_sub = word_prob(probs, tokenizer, best_sub)
    if p_sub is None:
        return None

    return p_sub - p_target


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def plot_violin(delta_p: dict[str, list[float]], out_path: str):
    methods = list(delta_p.keys())
    data = [delta_p[m] for m in methods]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.violinplot(data, positions=range(len(methods)), showmedians=True)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("ΔP = P(substitute|context) − P(original|context)")
    ax.set_title("CoInCo ΔP by Method")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_entry(entry, tokenizer, lm_model, backbone, caqe_model,
                   freq_probs, backbone_type, method_name, delta_p, device):
    sentence = entry["sentence"]
    target = entry["target"]
    substitutes = entry["substitutes"]

    token_pos = find_token_pos(tokenizer, sentence, target)
    if token_pos is None:
        return

    if backbone_type == "mlm":
        probs_lm = mlm_probs(lm_model, tokenizer, sentence, token_pos, device)
    else:
        probs_lm = ntp_probs(lm_model, tokenizer, sentence, token_pos, device)
    if probs_lm is not None:
        dp = compute_delta_p(probs_lm, tokenizer, target, substitutes)
        if dp is not None:
            delta_p[method_name].append(dp)

    probs_caqe = caqe_probs(caqe_model, backbone, tokenizer, sentence, token_pos, backbone_type, device)
    if probs_caqe is not None:
        dp = compute_delta_p(probs_caqe, tokenizer, target, substitutes)
        if dp is not None:
            delta_p["CAQE"].append(dp)

    dp = compute_delta_p(freq_probs, tokenizer, target, substitutes)
    if dp is not None:
        delta_p["Freq"].append(dp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["mlm", "ntp"], required=True)
    parser.add_argument("--model", required=True, help="model key from models.yaml")
    parser.add_argument("--ckpt", required=True, help="path to trained CAQE checkpoint (.pt)")
    parser.add_argument("--coinco", required=True, help="path to CoInCo txt file")
    parser.add_argument("--out", default="delta_p.png", help="output violin plot path")
    return parser.parse_args()


def main():
    args = parse_args()

    with open("configs/models.yaml") as f:
        models_cfg = yaml.safe_load(f)

    model_hf_id = models_cfg["backbone"][args.backbone][args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer and models: {model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Backbone for hidden extraction (base model, no LM head)
    backbone = AutoModel.from_pretrained(model_hf_id, dtype=torch.bfloat16).eval().to(device)

    # LM-head model for MLM/NTP baseline
    if args.backbone == "mlm":
        lm_model = AutoModelForMaskedLM.from_pretrained(model_hf_id, dtype=torch.bfloat16).eval().to(device)
    else:
        lm_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=torch.bfloat16).eval().to(device)

    # CAQE model
    backbone_cfg = AutoConfig.from_pretrained(model_hf_id)
    caqe_model = CAQE(backbone_cfg.hidden_size, backbone_cfg.vocab_size, models_cfg["caqe"])
    caqe_model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True)["model"])
    caqe_model.eval().to(device)

    entries = load_coinco(args.coinco)
    print(f"Loaded {len(entries)} CoInCo entries")

    freq_probs = build_freq_probs(tokenizer, entries, backbone_cfg.vocab_size)

    method_name = "MLM" if args.backbone == "mlm" else "NTP"
    delta_p: dict[str, list[float]] = {method_name: [], "CAQE": [], "Freq": []}

    for entry in entries:
        _process_entry(entry, tokenizer, lm_model, backbone, caqe_model,
                       freq_probs, args.backbone, method_name, delta_p, device)

    plot_violin(delta_p, args.out)
    for method, values in delta_p.items():
        if values:
            mean = sum(values) / len(values)
            print(f"{method}: n={len(values)}, mean ΔP={mean:.6f}")


if __name__ == "__main__":
    main()
