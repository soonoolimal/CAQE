#!/bin/bash
set -e
cd "$(dirname "$0")"

for backbone in bert roberta modernbert opt_1_3b llama31_8b llama32_3b; do
    python main.py analyze_rt --backbone_key "$backbone" --cuda 0
done

echo "All 6 extractions complete."
