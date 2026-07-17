import argparse
from pathlib import Path

import yaml

import plot
import scripts

_ROOT = Path(__file__).parent

with open(_ROOT / "configs" / "data.yaml") as _f:
    _BACKBONE_KEYS = tuple(yaml.safe_load(_f)["backbone"].keys())


def _load_credentials() -> tuple[str | None, str | None]:
    creds_path = _ROOT / "configs" / "_credentials.yaml"
    if not creds_path.exists():
        return None, None
    with open(creds_path) as f:
        creds = yaml.safe_load(f)
    return creds.get("wandb_api_key"), creds.get("wandb_entity")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("load_training_data")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--backbone_key", "--backbone", required=True, dest="backbone_key")
    train_parser.add_argument("--run_name", "--run", default=None, dest="run_name")
    train_parser.add_argument("--n_e", "--n", type=int, default=None, dest="n_e")
    train_parser.add_argument("--dec_dims", type=int, nargs="+", default=None, dest="dec_dims")
    train_parser.add_argument("--reinit_steps", "--re", type=int, default=None, dest="reinit_steps")
    train_parser.add_argument("--beta", type=float, default=None, dest="beta")
    train_parser.add_argument("--noise_reinit_ratio", type=float, default=None, dest="noise_reinit_ratio")
    train_parser.add_argument("--n_epochs", "--ep", type=int, default=None, dest="n_epochs")
    train_parser.add_argument("--diagnose", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument("--pca_wandb", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument("--cuda", type=int, default=None, metavar="GPU_ID")

    subparsers.add_parser("load_coinco")
    subparsers.add_parser("load_natural_stories")

    analyze_rt_parser = subparsers.add_parser("analyze_rt")
    analyze_rt_parser.add_argument("--backbone_key", "--backbone", required=True, dest="backbone_key")
    analyze_rt_parser.add_argument("--cuda", type=int, default=None, metavar="GPU_ID")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--backbone_key", "--backbone", required=True, dest="backbone_key")
    eval_parser.add_argument("--cuda", type=int, default=None, metavar="GPU_ID")

    plot_parser = subparsers.add_parser("plot_quant_error")
    plot_parser.add_argument("run_name")

    plot_eval_parser = subparsers.add_parser("plot_violin")
    plot_eval_parser.add_argument("--backbone_key", "--backbone", default=None, dest="backbone_key")

    plot_cross_parser = subparsers.add_parser("plot_cross")
    plot_cross_parser.add_argument("--backbone_key", "--backbone", default=None, dest="backbone_key")

    args = parser.parse_args()
    args.wandb_key, args.wandb_entity = _load_credentials()

    if args.command == "load_training_data":
        scripts.load_training_data()
    elif args.command == "load_coinco":
        scripts.load_coinco()
    elif args.command == "load_natural_stories":
        scripts.load_natural_stories()
    elif args.command == "analyze_rt":
        scripts.analyze_rt(args)
    elif args.command == "train":
        scripts.train_model(args)
    elif args.command == "eval":
        scripts.eval_model(args)
    elif args.command == "plot_quant_error":
        plot.plot_quant_error(_ROOT / "logs" / args.run_name)
    elif args.command == "plot_violin":
        for k in [args.backbone_key] if args.backbone_key else _BACKBONE_KEYS:
            args.backbone_key = k
            plot.plot_eval_violin(args)
    elif args.command == "plot_cross":
        for k in [args.backbone_key] if args.backbone_key else _BACKBONE_KEYS:
            args.backbone_key = k
            plot.plot_eval_cross(args)


if __name__ == "__main__":
    main()
