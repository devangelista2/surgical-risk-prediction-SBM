#!/bin/bash

# ==========================================
# MedModel Training Configuration
# ==========================================

# Data and Model Configs
DATA_CONFIG="./configs/preoperative.json"
MODEL_CONFIG="./configs/parameters.json"

# Models to train (leave empty "" to train all models in parameters.json)
MODELS="hgb,rf,svc,torch_mlp"

# Splitting Strategy: 'random', 'predefined', or 'temporal'
SPLIT_STRATEGY="temporal"
SPLIT_COLUMN="Split"
TEST_SIZE="0.2"
DATE_COLUMN="Date of surgery"

# ==========================================
# Run Training Script
# ==========================================

# Define the list of target variables
TARGETS=(
    "complications_30d"
    "Severe complication"
    "KPS_Discharge Worsened"
    "New neurological deficits"
)

echo "Starting MedModel Training Pipeline..."
echo "Split Strategy: $SPLIT_STRATEGY"
echo "=========================================="

# Loop through each target
for TARGET_COLUMN in "${TARGETS[@]}"; do
    
    # Update the output directory dynamically for each target
    OUTPUT_DIR="./outputs/$TARGET_COLUMN"
    
    echo ""
    echo ">>> Training for Target: $TARGET_COLUMN"
    echo ">>> Output Directory: $OUTPUT_DIR"

    # Build the command dynamically
    CMD="python ./src/train.py \
        --target \"$TARGET_COLUMN\" \
        --data_config \"$DATA_CONFIG\" \
        --model_config \"$MODEL_CONFIG\" \
        --output_folder \"$OUTPUT_DIR\" \
        --split_strategy \"$SPLIT_STRATEGY\" \
        --test_size $TEST_SIZE \
        --split_column \"$SPLIT_COLUMN\" \
        --date_column \"$DATE_COLUMN\""

    # Add models argument if it's not empty
    if [ -n "$MODELS" ]; then
        CMD="$CMD --models \"$MODELS\""
    fi

    # Execute the training command
    eval $CMD
    
    echo ">>> Finished training for: $TARGET_COLUMN"
    echo "------------------------------------------"
    
done

echo ""
echo "All experiments finished! Check the './outputs/' folder for results."