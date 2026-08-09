#!/usr/bin/env bash
set -euo pipefail

# Train the multivariate candidate magnitude stream in the two paper stages.

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
MASTER_PORT=${MASTER_PORT:-29522}
BATCH_SIZE=${BATCH_SIZE:-2}
RUN_STAGE=${RUN_STAGE:-all}

case "$RUN_STAGE" in
  all|stage1|stage2) ;;
  *) echo "RUN_STAGE must be one of: all, stage1, stage2" >&2; exit 2 ;;
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
  --llm_model "$LLM_MODEL"
  --llm_model_id "$LLM_MODEL_ID"
  --llm_dim "$LLM_DIM"
  --llm_layers "$LLM_LAYERS"
  --prompt_domain "$PROMPT_DOMAIN"
  --num_tokens "$MAGNITUDE_NUM_TOKENS"
  --prototype_mode "$MAGNITUDE_PROTOTYPE_MODE"
  --reprogramming_d_keys "$MAGNITUDE_REPROGRAMMING_D_KEYS"
  --loss JointMaskedAuxMAE
  --use_aux_loss
  --aux_loss_weight 0.1
  --regression_head_mlp
  --multivariate
  --channel_mixing
  --skip_test_during_training
  --skip_final_test
)

if [[ "$RUN_STAGE" == "all" || "$RUN_STAGE" == "stage1" ]]; then
  "${launch[@]}" "${common[@]}" \
    --train_epochs 50 \
    --learning_rate 0.001 \
    --model_comment iron_stage1_multivariate \
    --ablate_no_rmgm
fi

if [[ "$RUN_STAGE" == "all" || "$RUN_STAGE" == "stage2" ]]; then
  setting="long_term_forecast_Iron_${SEQ_LEN}_${PRED_LEN}_TimeLLM_custom_ftM_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_dm${D_MODEL}_nh8_el2_dl1_df${D_FF}_fc3_ebtimeF_Iron_Ore_Transport_Exp_0"
  stage1_dir=${STAGE1_DIR:-"$CHECKPOINT_ROOT/${setting}-iron_stage1_multivariate"}
  if [[ ! -d "$stage1_dir" ]]; then
    echo "Stage 1 checkpoint not found: $stage1_dir" >&2
    exit 2
  fi
  "${launch[@]}" "${common[@]}" \
    --train_epochs 20 \
    --learning_rate 0.00001 \
    --model_comment iron_stage2_multivariate \
    --load_ckpt_dir "$stage1_dir"
fi
