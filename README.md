# CAQE: Cognitive Aligned Vector Quantised Entropy

## Data

Preprocesses text and extracts hidden vectors for all backbones.

```bash
python main.py load_data
```

Preprocessed sentences are cached as JSONL in `data/train/preprocessed/`.
Hidden vectors are saved as chunked `.pt` files in `data/train/hidden_vecs/{model_key}/`, skipping any steps whose output already exists.

```
data/train/hidden_vecs/{model_key}/
├── chunk_0000.pt                                                     # hidden vectors + token ids, 10000 samples per chunk
├── chunk_0001.pt
├── ...
├── metadata.yaml                                                     # h_dim, vocab_size
└── kmeans_s{seed}_ne{n_e}_kms{km_seed}_ni{n_init}_bs{batch_size}.pt  # Stage 1 K-Means cache
```

## Training

Trains a VQCAE on hidden vectors extracted from a given backbone.

```bash
python main.py train --model MODEL [--run NAME] [--re STEPS] [--ep EPOCHS]
```

| Argument | Default | Description |
|---|---|---|
| `--model_key` (`--model`) | required | Backbone key: `bert`, `roberta`, `modernbert`, `llama32_3b`, `llama31_8b`, `opt_1_3b` |
| `--run_name` (`--run`) | — | Suffix appended to run name |
| `--reinit_steps` (`--re`) | `500` | Override `train.yaml` `reinit_steps` |
| `--n_epochs` (`--ep`) | `100` | Override `train.yaml` `n_epochs` |
| `--pca_local` | `False` | Save PCA snapshots locally (`pca/` dir created only when enabled) |
| `--pca_wandb` / `--no-pca_wandb` | `True` | Log PCA snapshots to wandb |
| `--diagnose` / `--no-diagnose` | `True` | Run diagnostic epoch before training to log `diagnostic/quant_error` per step |

The run name is formatted as `{model_key}_{run_name}_re{reinit_steps}_ep{n_epochs}_{MMDDHHMM}`.

Outputs are saved to `logs/{run_name}/`.

```
logs/{run_name}/
├── checkpoints/
│   ├── {run_name}_best{epoch}.pt  # best checkpoint
│   └── {run_name}_epoch{N}.pt     # interval checkpoints
├── pca/
│   └── epoch_{N}_step_{...}.png   # local PCA snapshots
├── diagnostic_quant_error.csv     # per-step quant_error from diagnostic epoch
└── {run_name}_best{epoch}.csv     # epoch-level metrics
```

Training is tracked on wandb under the project set in `configs/train.yaml`.

### Diagnostic Phase

Before training begins, one vanilla epoch (all losses active, no pre-reinit adjustment) is run with the Stage 1 codebook to log `diagnostic/quant_error` (mean distance from z_e to nearest codebook entry) per step. This curve reveals when codebook collapse begins, providing a quantitative basis for choosing `reinit_steps`. Use `--no-diagnose` to skip once `reinit_steps` is already determined.

### Reinitialization Stages

| Stage | Trigger | Description |
|---|---|---|
| Stage 1 | Start | K-Means codebook init over all train z_e (random encoder) |
| Stage 2 | Step `reinit_steps` | K-Means codebook reinit over all train z_e (partially trained encoder), with codebook optimizer(Adam) state reset |
| Stage 3 | Each epoch (first `noise_reinit_ratio * n_epochs` epochs) | Dead code reinit with gaussian noise centered at alive entry mean, with noise scale decaying linearly to 0 |

During steps 0 to `reinit_steps`-1 (pre-reinit), `cbook_loss` is excluded from backprop, and `step/cbook_active` in wandb marks the phase boundary (0: pre-reinit, 1: post-reinit).
