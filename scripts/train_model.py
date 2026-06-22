import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from core.model import VQCAE
from core.trainer import Trainer
from data.loader import make_loader, split_chunks

_ROOT = Path(__file__).parents[1]
_KST = timezone(timedelta(hours=9))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_configs(model_key: str) -> tuple[dict, dict, Path]:
    with open(_ROOT / "configs" / "train.yaml") as f:
        train_cfg = yaml.safe_load(f)
    with open(_ROOT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    chunk_dir = _ROOT / "data" / "train" / "hidden_vecs" / model_key
    with open(chunk_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    model_cfg["h_dim"] = metadata["h_dim"]
    model_cfg["vocab_size"] = metadata["vocab_size"]

    return train_cfg, model_cfg, chunk_dir


def train_model(args):
    train_cfg, model_cfg, chunk_dir = load_configs(args.model_key)
    if args.reinit_steps is not None:
        train_cfg["reinit_steps"] = args.reinit_steps
    if args.n_epochs is not None:
        train_cfg["n_epochs"] = args.n_epochs
    train_cfg["pca_local"] = args.pca_local
    train_cfg["pca_wandb"] = args.pca_wandb
    if args.diagnose is not None:
        train_cfg["run_diagnostic"] = args.diagnose
    set_seed(train_cfg["seed"])

    train_chunks, val_chunks = split_chunks(chunk_dir, train_cfg["val_ratio"])
    train_loader = make_loader(train_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=True)
    val_loader = make_loader(val_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQCAE(
        model_cfg,
        km_n_init=train_cfg["km_n_init"],
        km_batch_size=train_cfg["km_batch_size"],
        km_seed=train_cfg["km_seed"],
    ).to(device)

    timestamp = datetime.now(_KST).strftime("%m%d%H%M")
    re_tag = f"_re{train_cfg['reinit_steps']}_ep{train_cfg['n_epochs']}"
    run_name = f"{args.model_key}_{args.run_name}{re_tag}_{timestamp}" if args.run_name else f"{args.model_key}{re_tag}_{timestamp}"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        device=device,
        run_name=run_name,
        chunk_dir=chunk_dir,
    )
    trainer.train()
