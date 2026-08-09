#!/usr/bin/env bash
set -euo pipefail

# Train the event stream and the final recorded history branch.

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
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
NUM_PROCESSES=${NUM_PROCESSES:-2}
MASTER_PORT=${MASTER_PORT:-29543}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-10}
RUN_STAGE=${RUN_STAGE:-all}

case "$RUN_STAGE" in
  all|stage1|stage2|refine) ;;
  *) echo "RUN_STAGE must be one of: all, stage1, stage2, refine" >&2; exit 2 ;;
esac

if [[ ! -f "$DATA_ROOT/$DATA_FILE" ]]; then
  echo "Data file not found: $DATA_ROOT/$DATA_FILE" >&2
  exit 2
fi

launch=("$ACCELERATE_BIN" launch --mixed_precision bf16 --num_processes "$NUM_PROCESSES" --main_process_port "$MASTER_PORT")
if (( NUM_PROCESSES > 1 )); then
  launch+=(--multi_gpu)
fi

common=(
  "$ROOT/run_main.py"
  --task_name long_term_forecast
  --is_training 1
  --root_path "$DATA_ROOT"
  --data_path "$DATA_FILE"
  --checkpoints "$CHECKPOINT_ROOT"
  --model_id "Iron_${SEQ_LEN}_${PRED_LEN}"
  --model TimeLLM
  --data custom
  --seed "$SEED"
  --features M
  --seq_len "$SEQ_LEN"
  --label_len "$LABEL_LEN"
  --pred_len "$PRED_LEN"
  --factor 3
  --enc_in "$CHANNELS"
  --dec_in "$CHANNELS"
  --c_out "$CHANNELS"
  --des Iron_Ore_Transport_Exp
  --itr 1
  --d_model "$D_MODEL"
  --d_ff "$D_FF"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --llm_model "$LLM_MODEL"
  --llm_model_id "$LLM_MODEL_ID"
  --llm_dim "$LLM_DIM"
  --llm_layers "$LLM_LAYERS"
  --prompt_domain "$PROMPT_DOMAIN"
  --loss JointMaskedAuxMAE
  --use_aux_loss
  --aux_loss_weight 0.1
  --skip_test_during_training
  --skip_final_test
)

if [[ "$RUN_STAGE" == "all" || "$RUN_STAGE" == "stage1" ]]; then
  "${launch[@]}" "${common[@]}" \
    --train_epochs 50 \
    --learning_rate 0.001 \
    --model_comment iron_stage1_linear \
    --ablate_no_rmgm
fi

if [[ "$RUN_STAGE" == "all" || "$RUN_STAGE" == "stage2" ]]; then
  setting="long_term_forecast_Iron_${SEQ_LEN}_${PRED_LEN}_TimeLLM_custom_ftM_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_dm${D_MODEL}_nh8_el2_dl1_df${D_FF}_fc3_ebtimeF_Iron_Ore_Transport_Exp_0"
  stage1_dir=${STAGE1_DIR:-"$CHECKPOINT_ROOT/${setting}-iron_stage1_linear"}
  if [[ ! -d "$stage1_dir" ]]; then
    echo "Stage 1 checkpoint not found: $stage1_dir" >&2
    exit 2
  fi
  "${launch[@]}" "${common[@]}" \
    --train_epochs 20 \
    --learning_rate 0.00001 \
    --model_comment iron_stage2_linear_direct_nt1000_dk32 \
    --prototype_mode "$EVENT_PROTOTYPE_MODE" \
    --num_tokens "$EVENT_NUM_TOKENS" \
    --reprogramming_d_keys "$EVENT_REPROGRAMMING_D_KEYS" \
    --load_ckpt_dir "$stage1_dir"
fi

if [[ "$RUN_STAGE" == "all" || "$RUN_STAGE" == "refine" ]]; then
  setting="long_term_forecast_Iron_${SEQ_LEN}_${PRED_LEN}_TimeLLM_custom_ftM_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_dm${D_MODEL}_nh8_el2_dl1_df${D_FF}_fc3_ebtimeF_Iron_Ore_Transport_Exp_0"
  base_dir=${BASE_EVENT_DIR:-"$CHECKPOINT_ROOT/${setting}-iron_stage2_linear_direct_nt1000_dk32"}
  if [[ ! -d "$base_dir" ]]; then
    echo "Base event checkpoint not found: $base_dir" >&2
    exit 2
  fi
  "${launch[@]}" "${common[@]}" \
    --train_epochs 10 \
    --patience 4 \
    --learning_rate 0.0002 \
    --model_comment iron_stage2_linear_direct_nt1000_dk32_state_event_v2_window_aux \
    --prototype_mode "$EVENT_PROTOTYPE_MODE" \
    --num_tokens "$EVENT_NUM_TOKENS" \
    --reprogramming_d_keys "$EVENT_REPROGRAMMING_D_KEYS" \
    --state_event_gate \
    --state_event_gate_dim 64 \
    --state_event_feature_set v2 \
    --window_occurrence_head \
    --window_occurrence_head_dim 64 \
    --window_horizons 7,14,30,48 \
    --window_aux_loss_weight 0.05 \
    --window_aux_pos_weight 1.0 \
    --window_aux_neg_weight 1.0 \
    --aux_loss_type asym_focal \
    --aux_pos_weight 1.2 \
    --aux_neg_weight 1.0 \
    --aux_focal_gamma 1.0 \
    --aux_focal_gamma_neg 2.0 \
    --history_delta_anchor_weight 0.03 \
    --history_delta_mean_weight 0.03 \
    --history_delta_anchor_margin 0.10 \
    --freeze_except_history_gate \
    --save_last_checkpoint \
    --load_ckpt_dir "$base_dir"
fi
