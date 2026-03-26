#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere:
#   bash vis/run_vis_attn_mamba_tokens_v4.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIS_DIR="${ROOT}/vis/attn_mamba_final_v4"
mkdir -p "${VIS_DIR}"

# IAM (td_stride=4)  -> uses default images: test_0.png, test_1.png
python3 "${ROOT}/vis/vis_attn_mamba_tokens.py" \
  --vis-dir "${VIS_DIR}" \
  --block-idx -1 \
  --exp-name IAM_A0_noAniso \
  --td-stride 4 \
  --no-aniso \
  --strict-load \
  IAM

# LAM (td_stride=8)  -> uses default images: 002_02_00.jpg, 002_02_01.jpg
python3 "${ROOT}/vis/vis_attn_mamba_tokens.py" \
  --vis-dir "${VIS_DIR}" \
  --block-idx -1 \
  --exp-name LAM_A0_noAniso \
  --td-stride 8 \
  --no-aniso \
  --strict-load \
  LAM

# READ (td_stride=8) -> uses default images: Seite0001_1.png, Seite0001_2.png
python3 "${ROOT}/vis/vis_attn_mamba_tokens.py" \
  --vis-dir "${VIS_DIR}" \
  --block-idx -1 \
  --exp-name READ_A0_noAniso \
  --td-stride 8 \
  --no-aniso \
  --strict-load \
  READ

echo "[DONE] Saved figures to: ${VIS_DIR}"
