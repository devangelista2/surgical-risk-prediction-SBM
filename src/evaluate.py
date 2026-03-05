import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Fix: suppress warnings for cleaner console output
warnings.filterwarnings("ignore")

# Add src to sys.path to allow joblib to resolve custom classes (e.g., TorchMLPClassifier)
SRC_DIR = Path(__file__).parent.absolute()
if SRC_DIR.name != "src":
    SRC_DIR = SRC_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    import joblib
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: Missing dependencies. Please install: pandas, joblib, scikit-learn, numpy")
    sys.exit(1)

# Default configuration from training script
DEFAULT_TARGETS = [
    "complications_30d",
    "Severe complication",
    "KPS_Discharge Worsened",
    "New neurological deficits",
]
DEFAULT_MODELS = ["hgb", "rf", "svc", "torch_mlp"]


def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}".center(60))
    print("=" * 60)


def evaluate():
    parser = argparse.ArgumentParser(
        description="MedModel Evaluation Utility: Predict multiple targets across multiple models."
    )
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Input JSON string. e.g., '{\"Age\": 50, \"Pre-Op KPS\": 80}'",
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Path to a JSON file containing the input dictionary or list of dictionaries.",
    )

    # Configuration options
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated list of target variables.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of model names.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Base directory for trained model weights.",
    )

    args = parser.parse_args()

    # 1. Load Input Data
    try:
        if args.input:
            raw_data = json.loads(args.input)
        else:
            with open(args.input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        
        input_list = raw_data if isinstance(raw_data, list) else [raw_data]
        df_input = pd.DataFrame(input_list)
    except Exception as e:
        print(f"Error loading input data: {e}")
        sys.exit(1)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    output_base = Path(args.output_dir)

    print_header("MedModel - Multi-Model Prediction")
    print(f"Samples: {len(df_input)}")
    print(f"Targets: {len(targets)}")
    print(f"Models:  {len(models)}")
    print("-" * 60)

    # 2. Iterate over Targets and Models
    for target in targets:
        print(f"\n[TARGET] {target}")
        
        for model_name in models:
            model_path = output_base / target / model_name / "pipeline.joblib"
            
            if not model_path.exists():
                print(f"  - {model_name:12}: [NOT FOUND] at {model_path}")
                continue

            try:
                # Load pipeline
                # Note: src path is in sys.path, so custom classes should resolve
                pipeline = joblib.load(model_path)
                
                # Run prediction
                preds = pipeline.predict(df_input)
                
                # Check for probabilities
                probs = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        probs = pipeline.predict_proba(df_input)
                    except:
                        pass

                # Display results
                for i, pred in enumerate(preds):
                    sample_prefix = f"Sample {i+1} | " if len(df_input) > 1 else ""
                    
                    confidence_str = ""
                    if probs is not None:
                        try:
                            # Identify the confidence of the predicted class
                            classes = pipeline.classes_
                            pred_idx = np.where(classes == pred)[0][0]
                            conf = probs[i][pred_idx]
                            confidence_str = f" (Conf: {conf:.1%})"
                        except:
                            pass
                    
                    print(f"  - {model_name:12}: {sample_prefix}{pred}{confidence_str}")

            except Exception as e:
                print(f"  - {model_name:12}: [ERROR] {str(e)}")

    print("\n" + "=" * 60)
    print(" Evaluation Complete.".center(60))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate()
