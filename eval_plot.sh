#!/bin/bash
set -e
cd "$(dirname "$0")"

python main.py plot_violin
python main.py plot_cross

echo "All plots saved."
