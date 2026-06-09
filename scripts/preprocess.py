"""
Entry point for the CAQE preprocessing pipeline.

Downloads datasets from HuggingFace, preprocesses text, and extracts LLM hidden vectors.
Specify a dataset and backbone to prepare the training data for that combination.

Output:
    data/train/preprocessed/{dataset}.jsonl
    data/train/hidden_vecs/{dataset}/{model_key}/chunk_XXXX.pt

Usage (from project root):
    python scripts/preprocess.py --dataset gutenberg --backbone mlm
    python scripts/preprocess.py --dataset opensubtitles --backbone ntp
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data.extract_hidden import MLMHiddenExtractor, NTPHiddenExtractor
from data.load_corpus import GutenbergLoader, OpenSubtitlesLoader


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_models(backbone: str, models_cfg: dict) -> list[tuple[str, str, str]]:
    return [(key, hf_id, backbone) for key, hf_id in models_cfg["backbone"][backbone].items()]


def main():
    parser = argparse.ArgumentParser(
        description="CAQE preprocessing: corpus loading -> hidden vector extraction"
    )
    parser.add_argument("--dataset", choices=["gutenberg", "opensubtitles"], required=True)
    parser.add_argument("--backbone", choices=["mlm", "ntp"], required=True)
    args = parser.parse_args()

    data_cfg = load_config("configs/data.yaml")
    pre_cfg = load_config("configs/preprocess.yaml")
    models_cfg = load_config("configs/models.yaml")

    # ─── Step 1. Load Corpus
    # loads from local JSONL if available, otherwise downloads from HuggingFace and saves
    if args.dataset == "gutenberg":
        loader = GutenbergLoader(data_cfg["gutenberg"], pre_cfg)
    else:
        loader = OpenSubtitlesLoader(data_cfg["opensubtitles"], pre_cfg)
    texts = loader.load()

    # shuffle sentences so chunks contain random sentences,
    # making the tail-based train/val split representative (fixed seed keeps it consistent across backbones)
    random.seed(pre_cfg["shuffle_seed"])
    random.shuffle(texts)

    # ─── Step 2. Extract Hidden Vectors
    # runs each model in the backbone sequentially and saves hidden vectors as chunks
    models = resolve_models(args.backbone, models_cfg)
    out_base = Path(pre_cfg["output_dir"]) / pre_cfg["hidden_vec_dir"] / args.dataset

    for model_key, hf_id, backbone_type in models:
        print(f"\n{'='*60}")
        print(f"  [{backbone_type.upper()}] {model_key}  ×  {args.dataset}")
        print(f"  Total sentences: {len(texts):,}")
        print(f"{'='*60}")

        if backbone_type == "mlm":
            extractor = MLMHiddenExtractor(model_key, hf_id, pre_cfg)
        else:
            extractor = NTPHiddenExtractor(model_key, hf_id, pre_cfg)

        extractor.extract(texts, out_base)


if __name__ == "__main__":
    main()
