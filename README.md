# Cognitive Alignment Quantization-Based Entropy

## Preprocess

```bash
bash preprocess.sh [GPU_ID=0]
```

## Train

```bash
bash train.sh [N_E=8000] [GPU_ID=0]
```

EMA and dead code reinitialization are enabled by default. CLI arguments cover the primary experimental options (`--n_e`/`--ne`, `--no_ema`, `--no_reinit`). All other hyperparameters are managed in `configs/`.

Or run individually:

```bash
python -m train.run --backbone [mlm|ntp] --model [MODEL] [--n_e N_E] [--no_ema] [--no_reinit]
python -m train.run --backbone mlm --model bert
python -m train.run --backbone mlm --model roberta
python -m train.run --backbone mlm --model modernbert
python -m train.run --backbone ntp --model opt_1.3b
python -m train.run --backbone ntp --model llama3_3b
python -m train.run --backbone ntp --model llama31_8b
```
