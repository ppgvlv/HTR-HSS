#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   bash vis/run_bench_seq_len_time.sh
#
# It will write outputs into ./vis/
#
# You can override AMP (fp16 autocast) by adding:  --amp
# Example:
#   bash vis/run_bench_seq_len_time.sh --amp

EXTRA_ARGS="${*:-}"

python3 vis/bench_seq_len_time.py \
  --vis-dir vis \
  --device cuda \
  --batch-size 1 \
  --height 64 \
  --w-min 64 \
  --w-max 2048 \
  --w-step 64 \
  --warmup 10 \
  --iters 30 \
  ${EXTRA_ARGS}
