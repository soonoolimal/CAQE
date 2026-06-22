import argparse

from scripts.load_data import load_data
from scripts.train_model import train_model


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("load_data")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model_key", "--model", required=True, dest="model_key")
    train_parser.add_argument("--run_name", "--run", default=None, dest="run_name")
    train_parser.add_argument("--reinit_steps", "--re", type=int, default=None, dest="reinit_steps")
    train_parser.add_argument("--n_epochs", "--ep", type=int, default=None, dest="n_epochs")
    train_parser.add_argument("--pca_local", action=argparse.BooleanOptionalAction, default=False)
    train_parser.add_argument("--pca_wandb", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--diagnose", action=argparse.BooleanOptionalAction, default=None)

    args = parser.parse_args()

    if args.command == "load_data":
        load_data()
    elif args.command == "train":
        train_model(args)


if __name__ == "__main__":
    main()
