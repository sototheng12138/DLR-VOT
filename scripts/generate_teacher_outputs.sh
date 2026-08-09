#!/usr/bin/env bash
set -euo pipefail

# Generate Chronos-2 teacher outputs from the training split only.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAPER_CONFIG=${PAPER_CONFIG:-"$ROOT/configs/paper.env"}
if [[ ! -f "$PAPER_CONFIG" ]]; then
  echo "Configuration file not found: $PAPER_CONFIG" >&2
  exit 2
fi
# shellcheck source=../configs/paper.env
source "$PAPER_CONFIG"
DATA_ROOT=${DATA_ROOT:-"$ROOT/dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/checkpoints"}
PYTHON_BIN=${PYTHON_BIN:-python}
CHRONOS_MODEL_ID=${CHRONOS_MODEL_ID:-autogluon/chronos-2-small}
DEVICE_MAP=${DEVICE_MAP:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-float32}
BATCH_SIZE=${BATCH_SIZE:-8}

if [[ ! -f "$DATA_ROOT/$DATA_FILE" ]]; then
  echo "Data file not found: $DATA_ROOT/$DATA_FILE" >&2
  exit 2
fi

"$PYTHON_BIN" "$ROOT/scripts/Eval_Iron_chronos_zero_shot.py" \
  --root_path "$DATA_ROOT" \
  --data_path "$DATA_FILE" \
  --split train \
  --seq_len "$SEQ_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --features M \
  --model_id "$CHRONOS_MODEL_ID" \
  --device_map "$DEVICE_MAP" \
  --torch_dtype "$TORCH_DTYPE" \
  --batch_size "$BATCH_SIZE" \
  --out_dir "$OUTPUT_DIR" \
  --save_pred_true \
  --pred_tag chronos2_small_unclipped_train \
  --save_pred_true_space original
