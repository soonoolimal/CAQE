"""
Entry point for CAQE model training.

Usage (from project root):
    python scripts/train.py --dataset gutenberg --backbone mlm --model bert
    python scripts/train.py --dataset opensubtitles --backbone ntp --model llama3_3b
"""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig

from core.trainer import Trainer
from data.build_dataloader import make_dataloader, split_chunks
from models.caqe import CAQE


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAQE model")
    parser.add_argument("--dataset",  choices=["gutenberg", "opensubtitles"], required=True)
    parser.add_argument("--backbone", choices=["mlm", "ntp"], required=True)
    parser.add_argument("--model", required=True, help="model key from models.yaml")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models_cfg = load_config("configs/models.yaml")
    train_cfg = load_config("configs/train.yaml")
    pre_cfg = load_config("configs/preprocess.yaml")

    model_hf_id = models_cfg[args.backbone][args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    chunk_dir = Path(pre_cfg["output_dir"]) / pre_cfg["hidden_vec_dir"] / args.dataset / args.model
    train_chunks, val_chunks = split_chunks(chunk_dir, train_cfg["val_ratio"])
    train_loader = make_dataloader(train_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=True)
    val_loader = make_dataloader(val_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=False)

    # Model
    backbone_cfg = AutoConfig.from_pretrained(model_hf_id)
    model = CAQE(backbone_cfg.hidden_size, backbone_cfg.vocab_size, models_cfg).to(device)

    # Train
    run_name = f"{args.dataset}_{args.model}_{models_cfg['n_embeddings']}_caqe"
    trainer = Trainer(model, train_loader, val_loader, train_cfg, device, run_name)
    trainer.train()


if __name__ == "__main__":
    main()
