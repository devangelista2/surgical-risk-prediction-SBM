import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Model Imports
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from tqdm import tqdm

from nn.torch_ft_transformer import (
    TorchFTTransformerClassifier,
    TorchFTTransformerRegressor,
)
from nn.torch_mlp import TorchMLPClassifier, TorchMLPRegressor

# Import your existing utilities from preprocessing
from preprocessing import (
    CommaSeparatedMultiLabelBinarizer,
    UnixTimestampTransformer,
    infer_task_type,
    to_bool_if_binary,
)
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


def evaluate_val(pipeline, X_val, y_val, task_type):
    """Evaluates the model on the validation set for tuning purposes."""
    y_pred = pipeline.predict(X_val)

    if task_type == "continuous":
        # Minimize RMSE
        score = np.sqrt(mean_squared_error(y_val, y_pred))
        return score, "rmse"

    y_prob = (
        pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
    )

    if task_type == "binary" and y_prob is not None:
        pos_idx = 1 if y_prob.shape[1] > 1 else 0
        classes = getattr(pipeline.named_steps["model"], "classes_", [False, True])
        pos_label = classes[pos_idx]
        y_true_bin = (y_val == pos_label).astype(int)

        # Maximize ROC AUC
        score = roc_auc_score(y_true_bin, y_prob[:, pos_idx])
        return score, "roc_auc"

    else:
        # Maximize F1 Macro for categorical or fallback
        score = f1_score(y_val, y_pred, average="macro")
        return score, "f1_macro"


def main():
    parser = argparse.ArgumentParser(description="Temporal Grid Search for MedModel")
    parser.add_argument("--target", required=True, help="Target column.")
    parser.add_argument(
        "--data_config", default="data_config.json", help="Data configuration."
    )
    parser.add_argument(
        "--search_space", default="search_space.json", help="Grid search parameters."
    )
    parser.add_argument(
        "--output_file",
        default="best_parameters.json",
        help="Where to save the best configs.",
    )
    parser.add_argument(
        "--date_column",
        default="Date of surgery",
        help="Column used for temporal sorting.",
    )

    # Split percentages
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Held-out test set size (ignored during tuning).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set size (used to evaluate params).",
    )
    args = parser.parse_args()

    data_config = load_json(args.data_config)
    search_space = load_json(args.search_space)

    logger.info(f"Loading data from {data_config['input_file']}...")
    if str(data_config["input_file"]).endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_config["input_file"])
    else:
        try:
            df = pd.read_csv(data_config["input_file"], encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(data_config["input_file"], encoding="latin1")

    col_output = args.target
    df = df.dropna(subset=[col_output, args.date_column]).copy()

    task_type = infer_task_type(df[col_output])
    logger.info(f"Task: {task_type.upper()} | Target: {col_output}")

    if task_type == "binary":
        df[col_output] = to_bool_if_binary(df[col_output])

    # --- TEMPORAL SPLITTING (TRAIN / VAL / TEST) ---
    df["_temp_date"] = pd.to_datetime(df[args.date_column], errors="coerce")
    df = (
        df.dropna(subset=["_temp_date"])
        .sort_values(by="_temp_date")
        .drop(columns=["_temp_date"])
    )

    n_total = len(df)
    n_test = int(n_total * args.test_size)
    n_val = int(n_total * args.val_size)
    n_train = n_total - n_val - n_test

    # We hold out Test completely. We use Train to fit, Val to evaluate.
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]

    X_train, y_train = df_train, df_train[col_output]
    X_val, y_val = df_val, df_val[col_output]

    logger.info(
        f"Temporal Split -> Train: {n_train}, Val: {n_val}, Test (held out): {n_test}"
    )

    best_overall_params = {}

    for model_name, param_grid in search_space.items():
        logger.info(f"\n--- Tuning {model_name.upper()} ---")

        use_scaler = model_name in {
            "lr",
            "ridge",
            "svc",
            "torch_mlp",
            "torch_ft_transformer",
        }
        grid = list(ParameterGrid(param_grid))

        best_score = float("-inf") if task_type != "continuous" else float("inf")
        best_params = None
        metric_name = ""

        pbar = tqdm(grid, desc=f"Grid Search ({model_name})")
        for params in pbar:
            try:
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
                    X_val_t = pipeline.named_steps["preprocess"].transform(X_val)
                    # Pass the validation set directly to torch for early stopping / monitoring
                    pipeline.named_steps["model"].fit(
                        X_train_t, y_train, eval_set=(X_val_t, y_val)
                    )
                else:
                    pipeline.fit(X_train, y_train)

                score, metric_name = evaluate_val(pipeline, X_val, y_val, task_type)

                # Check if this is the best model so far
                if task_type == "continuous":
                    if score < best_score:  # Lower RMSE is better
                        best_score = score
                        best_params = params
                else:
                    if score > best_score:  # Higher AUC/F1 is better
                        best_score = score
                        best_params = params

                pbar.set_postfix({"Best Score": f"{best_score:.4f}"})

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")

        logger.info(
            f"Best {model_name} Params: {best_params} | Best Validation {metric_name.upper()}: {best_score:.4f}"
        )
        best_overall_params[model_name] = best_params

    # Save the best parameters to a JSON file
    out_path = Path(args.output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_overall_params, f, indent=4)

    logger.info(f"\nGrid Search Complete! Best parameters saved to '{out_path}'.")


if __name__ == "__main__":
    main()
