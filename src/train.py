import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from tqdm import tqdm

from nn.torch_ft_transformer import (
    TorchFTTransformerClassifier,
    TorchFTTransformerRegressor,
)
from nn.torch_mlp import TorchMLPClassifier, TorchMLPRegressor
from preprocessing import (
    CommaSeparatedMultiLabelBinarizer,
    UnixTimestampTransformer,
    infer_task_type,
    to_bool_if_binary,
)
from utils import vis
from utils.logger import logger

warnings.filterwarnings("ignore", category=UserWarning)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_preprocessor(
    input_features, cols_string, cols_date, cols_multi, use_scaler=True
):
    transformers = []

    numeric_cols = [
        c for c in input_features if c not in cols_string + cols_date + cols_multi
    ]

    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if use_scaler:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric_cols))

    if cols_string:
        steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("categorical", Pipeline(steps), cols_string))

    if cols_date:
        steps = [
            ("unix_ts", UnixTimestampTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if use_scaler:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("date", Pipeline(steps), cols_date))

    if cols_multi:
        steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("multilabel", CommaSeparatedMultiLabelBinarizer()),
        ]
        transformers.append(("multi", Pipeline(steps), cols_multi))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_model(model_name: str, task_type: str, params: dict):
    registry = {
        "binary": {
            "hgb": HistGradientBoostingClassifier,
            "rf": RandomForestClassifier,
            "lr": LogisticRegression,
            "svc": SVC,
            "torch_mlp": TorchMLPClassifier,
            "torch_ft_transformer": TorchFTTransformerClassifier,
        },
        "categorical": {
            "hgb": HistGradientBoostingClassifier,
            "rf": RandomForestClassifier,
            "lr": LogisticRegression,
            "svc": SVC,
            "torch_mlp": TorchMLPClassifier,
            "torch_ft_transformer": TorchFTTransformerClassifier,
        },
        "continuous": {
            "hgb": HistGradientBoostingRegressor,
            "rf": RandomForestRegressor,
            "ridge": Ridge,
            "svc": SVR,
            "torch_mlp": TorchMLPRegressor,
            "torch_ft_transformer": TorchFTTransformerRegressor,
        },
    }
    if model_name not in registry[task_type]:
        raise ValueError(f"Model '{model_name}' not valid for task '{task_type}'.")
    return registry[task_type][model_name](**params)


def evaluate_and_save(
    model_name: str, pipeline: Pipeline, X_test, y_test, task_type: str, out_dir: Path
):
    metrics = {}
    plot_data = {}  # Added to store data for combined plots
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    )

    if task_type == "continuous":
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

        vis.plot_regression_scatter(
            y_test,
            y_pred,
            f"Actual vs Predicted - {model_name.upper()}",
            model_dir / "actual_vs_predicted.png",
        )

    else:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        metrics["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        classes = getattr(pipeline.named_steps["model"], "classes_", np.unique(y_test))
        vis.plot_confusion_matrix(
            y_test,
            y_pred,
            classes,
            f"Confusion Matrix - {model_name.upper()}",
            model_dir / "confusion_matrix.png",
        )

        if y_prob is not None:
            if task_type == "binary":
                pos_idx = 1 if y_prob.shape[1] > 1 else 0
                y_prob_pos = y_prob[:, pos_idx]

                pos_label = classes[pos_idx]
                y_true_bin = (y_test == pos_label).astype(int)

                metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob_pos))
                metrics["average_precision"] = float(
                    average_precision_score(y_true_bin, y_prob_pos)
                )
                metrics["log_loss"] = float(log_loss(y_true_bin, y_prob_pos))

                vis.plot_roc_curve(
                    y_true_bin,
                    y_prob_pos,
                    f"ROC Curve - {model_name.upper()}",
                    model_dir / "roc_curve.png",
                )
                vis.plot_pr_curve(
                    y_true_bin,
                    y_prob_pos,
                    f"Precision-Recall Curve - {model_name.upper()}",
                    model_dir / "pr_curve.png",
                )

                # Save plot data for combined graphs
                plot_data["y_true_bin"] = y_true_bin
                plot_data["y_prob_pos"] = y_prob_pos

            elif task_type == "categorical":
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr")
                )
                metrics["log_loss"] = float(log_loss(y_test, y_prob))
                metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))

    joblib.dump(pipeline, model_dir / "pipeline.joblib")
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics, plot_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", required=True, help="Output column name to predict."
    )
    parser.add_argument(
        "--data_config", default="data_config.json", help="Data configuration"
    )
    parser.add_argument(
        "--model_config", default="parameters.json", help="Model hyperparameters"
    )
    parser.add_argument(
        "--output_folder", default="benchmark_output", help="Output directory"
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to train (e.g., hgb,rf,torch_mlp). If empty, trains all.",
    )

    parser.add_argument(
        "--split_strategy",
        choices=["random", "predefined", "temporal"],
        default="random",
        help="Strategy to split the train and test sets.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (for random and temporal).",
    )
    parser.add_argument(
        "--split_column",
        default="Split",
        help="Column name used for predefined split (e.g., 'Train' and 'Test').",
    )
    parser.add_argument(
        "--date_column",
        default="Date of surgery",
        help="Column name used for temporal split sorting.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_json(args.data_config)
    models_config = load_json(args.model_config)

    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
        models_to_train = {
            k: v for k, v in models_config.items() if k in selected_models
        }
    else:
        models_to_train = models_config

    input_file_path = data_config["input_file"]
    logger.info(f"Loading data from {input_file_path}...")

    if str(input_file_path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_file_path)
    else:
        try:
            df = pd.read_csv(input_file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(input_file_path, encoding="latin1")

    col_output = args.target
    if col_output not in df.columns:
        raise ValueError(f"Target column '{col_output}' not found in the dataset.")

    df = df.dropna(subset=[col_output]).copy()

    task_type = infer_task_type(df[col_output])
    logger.info(f"Target: '{col_output}' | Inferred task type: {task_type}")

    if task_type == "binary":
        df[col_output] = to_bool_if_binary(df[col_output])

    y = df[col_output]

    logger.info(f"Applying '{args.split_strategy}' split strategy...")

    if args.split_strategy == "predefined":
        if args.split_column not in df.columns:
            raise ValueError(
                f"Predefined split failed: column '{args.split_column}' not found."
            )

        split_col = df[args.split_column].astype(str).str.lower()
        train_mask = split_col.str.contains("train")
        test_mask = split_col.str.contains("test")

        X_train, X_test = df[train_mask], df[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

    elif args.split_strategy == "temporal":
        if args.date_column not in df.columns:
            raise ValueError(
                f"Temporal split failed: column '{args.date_column}' not found."
            )

        df_temp = df.copy()
        df_temp["_temp_date"] = pd.to_datetime(
            df_temp[args.date_column], errors="coerce"
        )
        df_temp = df_temp.dropna(subset=["_temp_date"]).sort_values(by="_temp_date")
        df_temp = df_temp.drop(columns=["_temp_date"])

        split_idx = int(len(df_temp) * (1 - args.test_size))
        X_train, X_test = df_temp.iloc[:split_idx], df_temp.iloc[split_idx:]
        y_train, y_test = X_train[col_output], X_test[col_output]

    else:  # random
        stratify = y if task_type != "continuous" else None
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=args.test_size, random_state=42, stratify=stratify
        )

    logger.info(f"Data split successful: {len(X_train)} Train, {len(X_test)} Test")

    experiment_metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": col_output,
        "task_type": task_type,
        "split_strategy": args.split_strategy,
        "split_config": {
            "test_size": (
                args.test_size if args.split_strategy != "predefined" else None
            ),
            "split_column": (
                args.split_column if args.split_strategy == "predefined" else None
            ),
            "date_column": (
                args.date_column if args.split_strategy == "temporal" else None
            ),
        },
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "models_trained": list(models_to_train.keys()),
        "data_configuration": data_config,
        "model_hyperparameters": models_to_train,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=4)

    benchmark_results = []

    # Dictionaries to store data for combined plots
    all_models_y_prob = {}
    shared_y_true_bin = None

    pbar = tqdm(models_to_train.items(), desc="Overall Training Progress", unit="model")

    for model_name, params in pbar:
        pbar.set_postfix({"Current Model": model_name})
        start_time = perf_counter()

        try:
            use_scaler = model_name in {
                "lr",
                "ridge",
                "svc",
                "torch_mlp",
                "torch_ft_transformer",
            }

            preprocessor = build_preprocessor(
                data_config["input_features"],
                data_config["cols_string"],
                data_config["cols_date"],
                data_config["cols_multi"],
                use_scaler,
            )

            model = get_model(model_name, task_type, params)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

            if model_name in {"torch_mlp", "torch_ft_transformer"}:
                X_train_t = pipeline.named_steps["preprocess"].fit_transform(
                    X_train, y_train
                )
                X_eval_t = pipeline.named_steps["preprocess"].transform(X_test)
                pipeline.named_steps["model"].fit(
                    X_train_t, y_train, eval_set=(X_eval_t, y_test)
                )
            else:
                pipeline.fit(X_train, y_train)

            fit_seconds = round(perf_counter() - start_time, 3)

            metrics, plot_data = evaluate_and_save(
                model_name, pipeline, X_test, y_test, task_type, out_dir
            )

            row = {"model": model_name, "status": "ok", "fit_seconds": fit_seconds}
            row.update(
                {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
            )
            benchmark_results.append(row)

            # Save plotting data for the combined charts
            if "y_prob_pos" in plot_data:
                all_models_y_prob[model_name] = plot_data["y_prob_pos"]
                shared_y_true_bin = plot_data["y_true_bin"]

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            benchmark_results.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": str(e),
                    "fit_seconds": round(perf_counter() - start_time, 3),
                }
            )

    summary_df = pd.DataFrame(benchmark_results)
    summary_path = out_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Generate combined plots if applicable
    if task_type == "binary" and all_models_y_prob:
        vis.plot_combined_roc_curve(
            shared_y_true_bin,
            all_models_y_prob,
            f"Combined ROC Curve ({col_output})",
            out_dir / "combined_roc_curve.png",
        )
        vis.plot_combined_pr_curve(
            shared_y_true_bin,
            all_models_y_prob,
            f"Combined PR Curve ({col_output})",
            out_dir / "combined_pr_curve.png",
        )

    print("\n" + "=" * 50)
    logger.info(f"Benchmark complete! Summary and plots saved to {out_dir}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
