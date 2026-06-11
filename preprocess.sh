#!/bin/bash
set -e

DEVICE=${1:-0}

CUDA_VISIBLE_DEVICES=$DEVICE python -m train.preprocess --dataset gutenberg --backbone mlm
CUDA_VISIBLE_DEVICES=$DEVICE python -m train.preprocess --dataset opensubtitles --backbone mlm
CUDA_VISIBLE_DEVICES=$DEVICE python -m train.preprocess --dataset gutenberg --backbone ntp
CUDA_VISIBLE_DEVICES=$DEVICE python -m train.preprocess --dataset opensubtitles --backbone ntp
