#!/bin/bash

# ==========================================
# MedModel Prediction Wrapper
# ==========================================

# Default Input file
DEFAULT_INPUT="./sample_input.json"
OUTPUT_DIR="./outputs"

# Check if an input file was provided as first argument
INPUT_FILE=${1:-$DEFAULT_INPUT}

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    echo "Usage: ./experiments/predict.sh [path_to_json_input]"
    exit 1
fi

echo "Starting MedModel Prediction Pipeline..."
echo "Input File: $INPUT_FILE"
echo "=========================================="

# Run the evaluation script
python ./src/evaluate.py --input_file "$INPUT_FILE" --output_dir "$OUTPUT_DIR"

echo "=========================================="
echo "Prediction finished."
