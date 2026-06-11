"""
Entry point for CAQE model training.

Trains on the combined hidden vectors from all datasets in data.yaml,
with each dataset split independently at val_ratio before combining.

CLI arguments cover the primary experimental options (codebook size, EMA, dead code reinitialization).
All other hyperparameters (learning rate, batch size, EMA gamma, etc.) are managed in configs/.

EMA and dead code reinitialization are enabled by default. Pass --no_ema or --no_reinit to disable.

Usage (from project root):
    python -m train.run --backbone mlm --model bert
    python -m train.run --backbone mlm --model bert --n_e 1024 --no_ema --no_reinit
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoConfig

from data.build_dataloader import make_dataloader, split_chunks
from models.caqe import CAQE
from train.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CAQE model")
    parser.add_argument("--backbone", choices=["mlm", "ntp"], required=True)
    parser.add_argument("--model", required=True, help="model key from models.yaml")
    parser.add_argument("--n_e", "--ne", type=int, default=None, dest="n_e", help="codebook size (overrides models.yaml)")
    parser.add_argument("--no_ema", action="store_true", default=False, help="disable EMA codebook update (overrides models.yaml)")
    parser.add_argument("--no_reinit", action="store_true", default=False, help="disable dead code reinitialization (overrides train.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()

    models_cfg = load_config("configs/models.yaml")
    train_cfg = load_config("configs/train.yaml")
    pre_cfg = load_config("configs/preprocess.yaml")
    data_cfg = load_config("configs/data.yaml")

    if args.n_e is not None:
        models_cfg["caqe"]["vqvae"]["n_e"] = args.n_e
    if args.no_ema:
        models_cfg["caqe"]["vqvae"]["ema_gamma"] = None
    train_cfg["dead_code_reinit"] = not args.no_reinit

    model_hf_id = models_cfg["backbone"][args.backbone][args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: split each dataset independently, then combine
    hidden_vec_dir = Path(pre_cfg["output_dir"]) / pre_cfg["hidden_vec_dir"]
    chunk_dirs = [hidden_vec_dir / dataset / args.model for dataset in data_cfg]
    train_chunks, val_chunks = split_chunks(chunk_dirs, train_cfg["val_ratio"])
    train_loader = make_dataloader(train_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=True)
    val_loader = make_dataloader(val_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=False)

    # Model
    backbone_cfg = AutoConfig.from_pretrained(model_hf_id)
    model = CAQE(backbone_cfg.hidden_size, backbone_cfg.vocab_size, models_cfg["caqe"]).to(device)

    # Train
    ema_suffix = "" if models_cfg["caqe"]["vqvae"]["ema_gamma"] is not None else "_noema"
    reinit_suffix = "" if train_cfg["dead_code_reinit"] else "_noreinit"
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%m%d%H%M")
    run_name = f"{args.model}_{models_cfg['caqe']['vqvae']['n_e']}{ema_suffix}{reinit_suffix}_{timestamp}"
    trainer = Trainer(model, train_loader, val_loader, train_cfg, models_cfg, device, run_name)
    trainer.train()


if __name__ == "__main__":
    main()
