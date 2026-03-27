#!/bin/bash

python3 train.py \
  --exp-name IAM_A0_noAniso \
  --max-lr 1e-3 \
  --train-bs 32 \
  --val-bs 8 \
  --weight-decay 0.5 \
  --mask-ratio 0.4 \
  --max-span-length 8 \
  --img-size 512 64 \
  --total-iter 100000 \
  --embed-dim 128 \
  --encoder-depth 6 \
  --mlp-ratio 4.0 \
  --drop-path-rate 0.1 \
  --num-levels 4 \
  --channel-multiplier 1.5 \
  --td-stride 4 \
  --attn-every 3 \
  --attn-heads 4 \
  --attn-window 0 \
  --no-aniso \
  IAM
