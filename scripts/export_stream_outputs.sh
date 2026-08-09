#!/usr/bin/env bash
set -euo pipefail

# Export only the validation and test arrays needed by VOT. This script does
# not search thresholds or calibrate on the test split.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAPER_CONFIG=${PAPER_CONFIG:-"$ROOT/configs/paper.env"}
if [[ ! -f "$PAPER_CONFIG" ]]; then
  echo "Configuration file not found: $PAPER_CONFIG" >&2
  exit 2
fi
# shellcheck source=../configs/paper.env
source "$PAPER_CONFIG"
DATA_ROOT=${DATA_ROOT:-"$ROOT/dataset"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$ROOT/checkpoints"}
PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}
STREAM=${STREAM:-all}

case "$STREAM" in
  all|magnitude|event) ;;
  *) echo "STREAM must be one of: all, magnitude, event" >&2; exit 2 ;;
esac

if [[ ! -f "$DATA_ROOT/$DATA_FILE" ]]; then
  echo "Data file not found: $DATA_ROOT/$DATA_FILE" >&2
  exit 2
fi

common=(
  "$ROOT/run_eval.py"
  --device "$DEVICE"
  --task_name long_term_forecast
  --root_path "$DATA_ROOT"
  --data_path "$DATA_FILE"
  --checkpoints "$CHECKPOINT_ROOT"
  --model_id "Iron_${SEQ_LEN}_${PRED_LEN}"
  --model TimeLLM
  --data custom
  --features M
  --seq_len "$SEQ_LEN"
  --label_len "$LABEL_LEN"
  --pred_len "$PRED_LEN"
  --enc_in "$CHANNELS"
  --dec_in "$CHANNELS"
  --c_out "$CHANNELS"
  --d_model "$D_MODEL"
  --d_ff "$D_FF"
  --factor 3
  --des Iron_Ore_Transport_Exp
  --llm_model "$LLM_MODEL"
  --llm_model_id "$LLM_MODEL_ID"
  --llm_dim "$LLM_DIM"
  --llm_layers "$LLM_LAYERS"
  --prompt_domain "$PROMPT_DOMAIN"
  --itr 1
  --use_aux_head
  --record_gate_prob_only
  --aux_confidence_threshold 0
  --zero_threshold 0
  --save_diagnostics
  --strict_checkpoint_load
  --eval_batch_size 1
)

export_pair() {
  local val_tag=$1
  local test_tag=$2
  shift 2
  "$PYTHON_BIN" "${common[@]}" --eval_split val --output_tag "$val_tag" "$@"
  "$PYTHON_BIN" "${common[@]}" --eval_split test --output_tag "$test_tag" "$@"
}

if [[ "$STREAM" == "all" || "$STREAM" == "magnitude" ]]; then
  export_pair ci_val_full ci_test_full \
    --model_comment iron_stage2_multivariate \
    --prototype_mode "$MAGNITUDE_PROTOTYPE_MODE" \
    --num_tokens "$MAGNITUDE_NUM_TOKENS" \
    --reprogramming_d_keys "$MAGNITUDE_REPROGRAMMING_D_KEYS" \
    --regression_head_mlp \
    --multivariate \
    --channel_mixing
fi

if [[ "$STREAM" == "all" || "$STREAM" == "event" ]]; then
  export_pair dg_val_noZT_full dg_test_noZT_full \
    --model_comment iron_stage2_linear_direct_nt1000_dk32_state_event_v2_window_aux \
    --prototype_mode "$EVENT_PROTOTYPE_MODE" \
    --num_tokens "$EVENT_NUM_TOKENS" \
    --reprogramming_d_keys "$EVENT_REPROGRAMMING_D_KEYS" \
    --state_event_gate \
    --state_event_gate_dim 64 \
    --state_event_feature_set v2 \
    --window_occurrence_head \
    --window_occurrence_head_dim 64 \
    --window_horizons 7,14,30,48
fi
