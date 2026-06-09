# Cognitive Alignment Quantization-Based Entropy

## Preprocess

```bash
bash preprocess.sh [GPU_ID=0]
```

## Train

```bash
bash train.sh [N_E=8000] [GPU_ID=0]
```

CLI arguments cover the primary experimental options (`--n_e`/`--ne`, `--ema`, `--dead_code_reinit`/`--reinit`). All other hyperparameters are managed in `configs/`.

Or run individually:

```bash
python scripts/train.py --backbone [mlm|ntp] --model [MODEL] [--n_e N_E] [--ema] [--dead_code_reinit]
python scripts/train.py --backbone mlm --model bert
python scripts/train.py --backbone mlm --model roberta
python scripts/train.py --backbone mlm --model modernbert
python scripts/train.py --backbone ntp --model opt_1.3b
python scripts/train.py --backbone ntp --model llama3_3b
python scripts/train.py --backbone ntp --model llama31_8b
```
