import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from model import VQCAE
from train import Logger, Trainer
from train.loader import make_loader, split_chunks

_ROOT = Path(__file__).parents[1]
_KST = timezone(timedelta(hours=9))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_configs(backbone_key: str) -> tuple[dict, dict, Path]:
    with open(_ROOT / "configs" / "train.yaml") as f:
        train_cfg = yaml.safe_load(f)
    with open(_ROOT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    chunk_dir = _ROOT / "data" / "train" / "hidden_vecs" / backbone_key
    with open(chunk_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    model_cfg["h_dim"] = metadata["h_dim"]
    model_cfg["vocab_size"] = metadata["vocab_size"]

    return train_cfg, model_cfg, chunk_dir


def train_model(args):
    train_cfg, model_cfg, chunk_dir = load_configs(args.backbone_key)
    for key in ("n_e", "dec_dims"):
        if (val := getattr(args, key)) is not None:
            model_cfg[key] = val
    for key in ("reinit_steps", "beta", "noise_reinit_ratio", "n_epochs", "diagnose", "pca_wandb"):
        if (val := getattr(args, key)) is not None:
            train_cfg[key] = val
    set_seed(train_cfg["seed"])

    train_chunks, val_chunks = split_chunks(chunk_dir, train_cfg["val_ratio"])
    train_loader = make_loader(train_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=True)
    val_loader = make_loader(val_chunks, train_cfg["batch_size"], train_cfg["num_workers"], shuffle=False)

    if args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQCAE(model_cfg).to(device)

    timestamp = datetime.now(_KST).strftime("%m%d%H%M")
    re_tag = f"n{model_cfg['n_e']}_re{train_cfg['reinit_steps']}_ep{train_cfg['n_epochs']}"
    run_name = (
        f"{args.backbone_key}_{args.run_name}_{re_tag}_{timestamp}"
        if args.run_name
        else f"{args.backbone_key}_{re_tag}_{timestamp}"
    )

    logger = Logger(
        run_name=run_name,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        wandb_key=args.wandb_key,
        wandb_entity=args.wandb_entity,
    )
    trainer = Trainer(
        model=model,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        chunk_dir=chunk_dir,
        device=device,
        logger=logger,
    )

    trainer.train()
