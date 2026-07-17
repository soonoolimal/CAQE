#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p logs/_gpu

(
    for backbone in bert roberta modernbert; do
        for n_e in 1000 2000 4000 8000; do
            python main.py train --backbone_key "$backbone" --n_e "$n_e" --cuda 0
        done
    done
) > logs/_gpu/gpu0.log 2>/dev/null &  # 2>&1 causes tqdm progress bars to bloat the log file

(
    for backbone in opt_1_3b llama31_8b llama32_3b; do
        for n_e in 1000 2000 4000 8000; do
            python main.py train --backbone_key "$backbone" --n_e "$n_e" --cuda 1
        done
    done
) > logs/_gpu/gpu1.log 2>/dev/null &

# monitor:
#   tail -f logs/_gpu/gpu0.log
#   tail -f logs/_gpu/gpu1.log

wait
echo "All 24 runs complete."
rm -rf logs/_gpu
