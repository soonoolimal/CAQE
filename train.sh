#!/bin/bash

N_E=${1:-8000}
GPU=${2:-0}

CUDA_VISIBLE_DEVICES=$GPU python -m train.run --backbone mlm --model bert --n_e $N_E &&
CUDA_VISIBLE_DEVICES=$GPU python -m train.run --backbone mlm --model roberta --n_e $N_E &&
CUDA_VISIBLE_DEVICES=$GPU python -m train.run --backbone mlm --model modernbert --n_e $N_E &&
CUDA_VISIBLE_DEVICES=$GPU python -m train.run --backbone ntp --model opt_1.3b --n_e $N_E &&
CUDA_VISIBLE_DEVICES=$GPU python -m train.run --backbone ntp --model llama3_3b --n_e $N_E &&
CUDA_VISIBLE_DEVICES=$GPU python -m train.run --backbone ntp --model llama31_8b --n_e $N_E
