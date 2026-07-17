# Hyperparameters

## model.yaml

| Parameter | Default | Description |
|---|---|---|
| `n_e` | 8000 | Codebook size (number of codes) |
| `e_dim` | 256 | Code embedding dimension |
| `enc_h_dim` | 512 | Encoder hidden dimension |
| `dec_dims` | `[512]` | Decoder hidden dims (`e_dim -> dims[0] -> ... -> h_dim`), override per backbone via `--dec_dims` |
| `clf_h_dim` | 256 | Classifier hidden dimension |

## train.yaml

### Setup

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Global random seed |
| `num_workers` | 4 | DataLoader worker count |
| `batch_size` | 512 | Training batch size |
| `n_epochs` | 200 | Number of training epochs |
| `val_ratio` | 0.2 | Fraction of chunks held out for validation |

### Codebook Reinitialization

| Parameter | Default | Description |
|---|---|---|
| `km_seed` | 42 | K-Means random seed |
| `km_n_init` | 1 | Number of K-Means runs per reinit (best inertia kept, but 1 is sufficient since full data is used) |
| `km_batch_size` | 8192 | MiniBatchKMeans batch size per iteration |
| `reinit_steps` | 500 | Reinit 2 fires once when `global_step` reaches this value |
| `noise_reinit_ratio` | 0.1 | Reinit 3 is active for the first `ratio * n_epochs` epochs |
| `noise_init_scale` | 0.1 | Reinit 3 starting noise magnitude, decays linearly to 0 |

### Optimizer and Scheduler

| Parameter | Default | Description |
|---|---|---|
| `beta` | 1.0 | Commitment loss weight |
| `scheduler` | `"cosine"` | LR scheduler: `cosine` or `none` |
| `lr` | 3.0e-4 | Learning rate (3.0e-4 with `cosine`, 1.0e-4 with `none`) |
| `weight_decay` | 0.0 | Adam weight decay |
| `early_stopping` | true | Enable early stopping (only active when `scheduler` is `none`) |
| `patience` | 10 | Early stopping patience (epochs) |
| `min_delta` | 0.0 | Minimum validation loss improvement to reset patience |

### Diagnostic and W&B Logging

| Parameter | Default | Description |
|---|---|---|
| `wandb_project` | `"VQCAE-Training-Experiments"` | W&B project name |
| `diagnose` | true | Run one vanilla epoch before training to log `quant_error` for `reinit_steps` analysis |
| `diag_steps_ratio` | 1.5 | Diagnostic stops at `reinit_steps * ratio` (covers peak and early downslope) |
| `pca_wandb` | true | Upload PCA snapshots to W&B |
| `pre_reinit_pca_n` | 10 | Number of evenly-spaced PCA snapshots in (0, `reinit_steps`) |
| `pca_ze_ratio` | 1.5 | z_e sample count = `n_e * ratio` |

### Local Logging

| Parameter | Default | Description |
|---|---|---|
| `local_log_dir` | `"logs"` | Root directory for run logs |
| `ckpt_save_interval` | 10 | Checkpoint save interval (epochs) |
