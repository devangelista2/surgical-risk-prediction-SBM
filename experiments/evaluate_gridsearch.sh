#!/usr/bin/env bash
set -euo pipefail

# Edit only these variables
DATA_CONFIG="configs/preoperative_reduced.json"
GRIDSEARCH_DIR="configs/gridsearch"
OUTPUT_ROOT="outputs/gridsearch_eval"
TRAIN_SCRIPT="src/train.py"
TARGETS=(
  "complications_30d"
  "Severe complication"
  "KPS_Discharge Worsened"
  "New neurological deficits"
)

# Splitting Strategy: 'random', 'predefined', or 'temporal'
SPLIT_STRATEGY="temporal"
SPLIT_COLUMN="Split"
TEST_SIZE="0.2"
DATE_COLUMN="Date of surgery"

# FN-sensitive decision policy (binary tasks)
THRESHOLD_VAL_SIZE="0.2"
MIN_RECALL="0.80"
F_BETA="2.0"
FN_COST="1.0"
FP_COST="5.0"

mkdir -p "$OUTPUT_ROOT"

for target in "${TARGETS[@]}"; do
  best_params="$GRIDSEARCH_DIR/$target/best_parameters.json"

  if [[ ! -f "$best_params" ]]; then
    echo "[SKIP] $target -> missing $best_params"
    continue
  fi

  models_csv="$(python - "$best_params" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    d = json.load(f)
print(','.join([k for k, v in d.items() if v is not None]))
PY
)"

  echo "[RUN ] $target | models: $models_csv"
  python "$TRAIN_SCRIPT" \
    --target "$target" \
    --data_config "$DATA_CONFIG" \
    --model_config "$best_params" \
    --models "$models_csv" \
    --output_folder "$OUTPUT_ROOT/$target" \
    --feature_importance \
    --test_size $TEST_SIZE \
    --split_strategy "$SPLIT_STRATEGY" \
    --split_column "$SPLIT_COLUMN" \
    --date_column "$DATE_COLUMN" \
    --threshold_val_size $THRESHOLD_VAL_SIZE \
    --min_recall $MIN_RECALL \
    --f_beta $F_BETA \
    --fn_cost $FN_COST \
    --fp_cost $FP_COST
done

echo "Done."
