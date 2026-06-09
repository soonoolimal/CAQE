#!/bin/bash
set -e

DEVICE=${1:-0}

CUDA_VISIBLE_DEVICES=$DEVICE python scripts/preprocess.py --dataset gutenberg --backbone mlm
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/preprocess.py --dataset opensubtitles --backbone mlm
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/preprocess.py --dataset gutenberg --backbone ntp
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/preprocess.py --dataset opensubtitles --backbone ntp
