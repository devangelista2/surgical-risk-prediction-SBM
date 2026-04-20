#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# MedModel Training From Grid Search Outputs
# ==========================================

# Data config used for training
DATA_CONFIG="./configs/preoperative.json"

# Grid search output root containing per-target best_parameters.json
GRIDSEARCH_ROOT="./gridsearch/preoperative"

# Training output root
OUTPUT_ROOT="./outputs/preoperative_from_gridsearch"

# Training script
TRAIN_SCRIPT="./src/train.py"

# Splitting Strategy: 'random', 'predefined', or 'temporal'
SPLIT_STRATEGY="temporal"
SPLIT_COLUMN="Split"
TEST_SIZE="0.2"
DATE_COLUMN="Date of surgery"

# FN-sensitive decision policy (binary tasks)
THRESHOLD_VAL_SIZE="0.2"
MIN_RECALL="0.90"
F_BETA="2.0"
FN_COST="5.0"
FP_COST="1.0"

# Optional extras
FEATURE_IMPORTANCE="true"

# Define the list of target variables
TARGETS=(
    "complications_30d"
    "Severe complication"
    "KPS_Discharge Worsened"
    "New neurological deficits"
)

mkdir -p "$OUTPUT_ROOT"

echo "Starting MedModel Training Pipeline From Grid Search..."
echo "Grid Search Root: $GRIDSEARCH_ROOT"
echo "Split Strategy: $SPLIT_STRATEGY"
echo "=========================================="

for TARGET_COLUMN in "${TARGETS[@]}"; do
    BEST_PARAMS_FILE="$GRIDSEARCH_ROOT/$TARGET_COLUMN/best_parameters.json"
    OUTPUT_DIR="$OUTPUT_ROOT/$TARGET_COLUMN"

    if [[ ! -f "$BEST_PARAMS_FILE" ]]; then
        echo "[SKIP] $TARGET_COLUMN -> missing $BEST_PARAMS_FILE"
        continue
    fi

    MODELS="$(
        python - "$BEST_PARAMS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    params = json.load(f)

models = [model_name for model_name, model_params in params.items() if isinstance(model_params, dict)]
print(",".join(models))
PY
    )"

    if [[ -z "$MODELS" ]]; then
        echo "[SKIP] $TARGET_COLUMN -> no trainable models found in $BEST_PARAMS_FILE"
        continue
    fi

    echo ""
    echo ">>> Training for Target: $TARGET_COLUMN"
    echo ">>> Best Parameters: $BEST_PARAMS_FILE"
    echo ">>> Models: $MODELS"
    echo ">>> Output Directory: $OUTPUT_DIR"

    CMD="python \"$TRAIN_SCRIPT\" \
        --target \"$TARGET_COLUMN\" \
        --data_config \"$DATA_CONFIG\" \
        --model_config \"$BEST_PARAMS_FILE\" \
        --models \"$MODELS\" \
        --output_folder \"$OUTPUT_DIR\" \
        --split_strategy \"$SPLIT_STRATEGY\" \
        --test_size $TEST_SIZE \
        --split_column \"$SPLIT_COLUMN\" \
        --date_column \"$DATE_COLUMN\" \
        --threshold_val_size $THRESHOLD_VAL_SIZE \
        --min_recall $MIN_RECALL \
        --f_beta $F_BETA \
        --fn_cost $FN_COST \
        --fp_cost $FP_COST"

    if [ "$FEATURE_IMPORTANCE" = "true" ]; then
        CMD="$CMD --feature_importance"
    fi

    eval $CMD

    echo ">>> Finished training for: $TARGET_COLUMN"
    echo "------------------------------------------"
done

echo ""
echo "All experiments finished! Check '$OUTPUT_ROOT' for results."
