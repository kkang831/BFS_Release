#!/usr/bin/env bash
set -euo pipefail

: "${TRANS_VAE_PATH:?Set TRANS_VAE_PATH to the transparency VAE path or model ID}"
: "${BFS_CHECKPOINT:?Set BFS_CHECKPOINT to trained_weights.pt or its directory}"

BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-Fill-dev}"
INPUT_PATH="${INPUT_PATH:-./inputs/images}"
MASK_PATH="${MASK_PATH:-./inputs/masks}"
CAPTION_PATH="${CAPTION_PATH:-./inputs/captions}"
OUTPUT_PATH="${OUTPUT_PATH:-./results}"

python3 inference.py \
  --pretrained_model_name_or_path "${BASE_MODEL}" \
  --pretrained_trans_vae_path "${TRANS_VAE_PATH}" \
  --pretrained_lora_path "${BFS_CHECKPOINT}" \
  --input_path "${INPUT_PATH}" \
  --mask_path "${MASK_PATH}" \
  --foreground_caption_path "${CAPTION_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --return_recomposite
