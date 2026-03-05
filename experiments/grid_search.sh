#!/bin/bash

# ==========================================
# MedModel Grid Search Configuration
# ==========================================

# Data and Search Space Configs (Adjust paths if needed based on where you run the script)
DATA_CONFIG="./configs/preoperative.json"
SEARCH_SPACE="./configs/grid_search.json"

# Temporal Split Settings
DATE_COLUMN="Date of surgery"
TEST_SIZE="0.15"
VAL_SIZE="0.15"

# Define the list of target variables
TARGETS=(
    "Severe complication"
    "KPS_Discharge Worsened"
)

echo "Starting MedModel Grid Search Pipeline..."
echo "=========================================="

# Loop through each target
for TARGET_COLUMN in "${TARGETS[@]}"; do
    
    # Create the output directory for this target if it doesn't exist
    OUTPUT_DIR="./gridsearch/preoperative/$TARGET_COLUMN"
    mkdir -p "$OUTPUT_DIR"
    
    # Define where the best parameters file should be saved
    OUTPUT_FILE="$OUTPUT_DIR/best_parameters.json"
    
    echo ""
    echo ">>> Tuning for Target: $TARGET_COLUMN"
    echo ">>> Saving best parameters to: $OUTPUT_FILE"

    # Build the command dynamically
    CMD="python ./src/tune.py \
        --target \"$TARGET_COLUMN\" \
        --data_config \"$DATA_CONFIG\" \
        --search_space \"$SEARCH_SPACE\" \
        --output_file \"$OUTPUT_FILE\" \
        --date_column \"$DATE_COLUMN\" \
        --test_size $TEST_SIZE \
        --val_size $VAL_SIZE"

    # Execute the tuning command
    eval $CMD
    
    echo ">>> Finished tuning for: $TARGET_COLUMN"
    echo "------------------------------------------"
    
done

echo ""
echo "All grid search experiments finished! You can now run the train.sh script using these optimized parameters."