import argparse
import json
import warnings
from pathlib import Path

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
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, roc_auc_score
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


def build_preprocessor(input_features, cols_string, cols_date, cols_multi, use_scaler=True):
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


def get_registry():
    return {
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


def get_model(model_name: str, task_type: str, params: dict):
    registry = get_registry()
    if model_name not in registry[task_type]:
        raise ValueError(f"Model '{model_name}' not valid for task '{task_type}'.")
    return registry[task_type][model_name](**params)


def balanced_weights_from_y(y_series):
    y_np = np.asarray(y_series)
    classes, counts = np.unique(y_np, return_counts=True)
    n_samples = len(y_np)
    n_classes = len(classes)
    weights = {
        cls: float(n_samples / (n_classes * count))
        for cls, count in zip(classes, counts, strict=False)
    }
    return weights


def sample_weight_from_y(y_series):
    class_weights = balanced_weights_from_y(y_series)
    y_np = np.asarray(y_series)
    return np.asarray([class_weights[v] for v in y_np], dtype=float)


def normalize_class_weight_keys(class_weight: dict, task_type: str):
    normalized = {}
    for key, value in class_weight.items():
        new_key = key
        if task_type == "binary":
            if isinstance(key, str):
                lk = key.strip().lower()
                if lk in {"true", "1"}:
                    new_key = True
                elif lk in {"false", "0"}:
                    new_key = False
        normalized[new_key] = value
    return normalized


def apply_imbalance_strategy(model_name: str, task_type: str, params: dict, y_train):
    new_params = dict(params)
    if "class_weight" in new_params and isinstance(new_params["class_weight"], dict):
        new_params["class_weight"] = normalize_class_weight_keys(
            new_params["class_weight"], task_type
        )

    if task_type not in {"binary", "categorical"}:
        return new_params

    weights = balanced_weights_from_y(y_train)
    if not weights:
        return new_params

    if model_name in {"rf", "lr", "svc"} and "class_weight" not in new_params:
        new_params["class_weight"] = {k.item() if hasattr(k, "item") else k: v for k, v in weights.items()}

    if model_name in {"torch_mlp", "torch_ft_transformer"}:
        classes_sorted = sorted(weights.keys())
        if task_type == "binary" and "pos_weight" not in new_params:
            neg_label, pos_label = classes_sorted[0], classes_sorted[-1]
            neg_w = weights[neg_label]
            pos_w = weights[pos_label]
            if neg_w > 0:
                new_params["pos_weight"] = float(pos_w / neg_w)
        elif task_type == "categorical" and "class_weights" not in new_params:
            new_params["class_weights"] = [float(weights[c]) for c in classes_sorted]

    return new_params


def threshold_metrics(y_true_bin, y_prob_pos, threshold, beta, fn_cost, fp_cost):
    y_pred = (y_prob_pos >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    beta2 = beta * beta
    f_beta = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if (precision + recall)
        else 0.0
    )
    cost = fn_cost * fn + fp_cost * fp
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "npv": float(npv),
        "f_beta": float(f_beta),
        "cost": float(cost),
    }


def select_threshold(y_true_bin, y_prob_pos, min_recall, beta, fn_cost, fp_cost):
    thresholds = np.linspace(0.01, 0.99, 199)
    metrics = [
        threshold_metrics(y_true_bin, y_prob_pos, t, beta, fn_cost, fp_cost)
        for t in thresholds
    ]
    feasible = [m for m in metrics if m["recall"] >= min_recall]

    if feasible:
        best = max(
            feasible,
            key=lambda m: (m["precision"], m["f_beta"], -m["cost"], m["specificity"]),
        )
        best["meets_recall_constraint"] = True
        return best

    best = max(
        metrics,
        key=lambda m: (m["recall"], m["precision"], m["f_beta"], -m["cost"]),
    )
    best["meets_recall_constraint"] = False
    return best


def evaluate_val(
    pipeline,
    X_val,
    y_val,
    task_type,
    min_recall,
    beta,
    fn_cost,
    fp_cost,
):
    y_pred = pipeline.predict(X_val)

    if task_type == "continuous":
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        return {
            "sort_key": rmse,
            "metric_name": "rmse",
            "display_score": rmse,
            "details": {"rmse": rmse},
        }

    y_prob = (
        pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
    )

    if task_type == "binary" and y_prob is not None:
        pos_idx = 1 if y_prob.shape[1] > 1 else 0
        classes = getattr(pipeline.named_steps["model"], "classes_", [False, True])
        pos_label = classes[pos_idx]
        y_true_bin = (y_val == pos_label).astype(int).to_numpy()
        y_prob_pos = y_prob[:, pos_idx]

        try:
            auc = float(roc_auc_score(y_true_bin, y_prob_pos))
        except ValueError:
            auc = float("nan")

        op = select_threshold(y_true_bin, y_prob_pos, min_recall, beta, fn_cost, fp_cost)
        op["auc_roc"] = auc

        if op["meets_recall_constraint"]:
            sort_key = (2, op["precision"], op["f_beta"], auc if np.isfinite(auc) else -1, -op["cost"])
            return {
                "sort_key": sort_key,
                "metric_name": f"precision@recall>={min_recall:.2f}",
                "display_score": op["precision"],
                "details": op,
            }

        sort_key = (1, op["recall"], op["precision"], auc if np.isfinite(auc) else -1, -op["cost"])
        return {
            "sort_key": sort_key,
            "metric_name": f"max_recall_if_<{min_recall:.2f}",
            "display_score": op["recall"],
            "details": op,
        }

    f1m = float(f1_score(y_val, y_pred, average="macro"))
    return {
        "sort_key": f1m,
        "metric_name": "f1_macro",
        "display_score": f1m,
        "details": {"f1_macro": f1m},
    }


def is_better_eval(task_type, new_eval, best_eval):
    if best_eval is None:
        return True
    if task_type == "continuous":
        return new_eval["sort_key"] < best_eval["sort_key"]
    return new_eval["sort_key"] > best_eval["sort_key"]


def main():
    parser = argparse.ArgumentParser(description="Temporal Grid Search for MedModel")
    parser.add_argument("--target", required=True, help="Target column.")
    parser.add_argument("--data_config", default="data_config.json", help="Data configuration.")
    parser.add_argument("--search_space", default="search_space.json", help="Grid search parameters.")
    parser.add_argument("--output_file", default="best_parameters.json", help="Where to save the best configs.")
    parser.add_argument("--date_column", default="Date of surgery", help="Column used for temporal sorting.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Held-out test set size (ignored during tuning).")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set size (used to evaluate params).")
    parser.add_argument("--min_recall", type=float, default=0.90, help="Binary tuning constraint: minimum recall target.")
    parser.add_argument("--f_beta", type=float, default=2.0, help="Beta for F-beta during threshold optimization.")
    parser.add_argument("--fn_cost", type=float, default=5.0, help="Relative cost assigned to each false negative.")
    parser.add_argument("--fp_cost", type=float, default=1.0, help="Relative cost assigned to each false positive.")
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

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]

    X_train, y_train = df_train, df_train[col_output]
    X_val, y_val = df_val, df_val[col_output]

    logger.info(f"Temporal Split -> Train: {n_train}, Val: {n_val}, Test (held out): {n_test}")

    registry = get_registry()
    valid_models = set(registry[task_type].keys())

    best_overall_params = {}
    best_selection_details = {}

    for model_name, param_grid in search_space.items():
        logger.info(f"\n--- Tuning {model_name.upper()} ---")

        if model_name not in valid_models:
            logger.info(f"Skipping {model_name}: not valid for task '{task_type}'.")
            best_overall_params[model_name] = None
            best_selection_details[model_name] = {"status": "invalid_for_task"}
            continue

        use_scaler = model_name in {
            "lr",
            "ridge",
            "svc",
            "torch_mlp",
            "torch_ft_transformer",
        }
        grid = list(ParameterGrid(param_grid))

        best_params = None
        best_eval = None
        best_metric_name = ""
        fail_count = 0

        pbar = tqdm(grid, desc=f"Grid Search ({model_name})")
        for raw_params in pbar:
            try:
                params = apply_imbalance_strategy(model_name, task_type, raw_params, y_train)

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
                    X_train_t = pipeline.named_steps["preprocess"].fit_transform(X_train, y_train)
                    X_val_t = pipeline.named_steps["preprocess"].transform(X_val)
                    pipeline.named_steps["model"].fit(X_train_t, y_train, eval_set=(X_val_t, y_val))
                else:
                    fit_kwargs = {}
                    if model_name == "hgb" and task_type in {"binary", "categorical"}:
                        fit_kwargs["model__sample_weight"] = sample_weight_from_y(y_train)
                    pipeline.fit(X_train, y_train, **fit_kwargs)

                eval_result = evaluate_val(
                    pipeline,
                    X_val,
                    y_val,
                    task_type,
                    min_recall=args.min_recall,
                    beta=args.f_beta,
                    fn_cost=args.fn_cost,
                    fp_cost=args.fp_cost,
                )

                if is_better_eval(task_type, eval_result, best_eval):
                    best_eval = eval_result
                    best_metric_name = eval_result["metric_name"]
                    best_params = params

                pbar.set_postfix({"Best": f"{best_eval['display_score']:.4f}" if best_eval else "n/a"})
            except Exception as e:
                fail_count += 1
                logger.warning(f"Failed with params {raw_params}: {e}")

        detail = {
            "status": "ok" if best_params is not None else "failed",
            "metric_name": best_metric_name,
            "metric_value": best_eval["display_score"] if best_eval else None,
            "failed_trials": fail_count,
        }
        if best_eval and "details" in best_eval:
            detail.update(best_eval["details"])

        logger.info(
            f"Best {model_name} Params: {best_params} | "
            f"Best Validation {best_metric_name.upper() if best_metric_name else 'N/A'}: "
            f"{best_eval['display_score']:.4f}" if best_eval else f"Best {model_name} Params: None"
        )

        best_overall_params[model_name] = best_params
        best_selection_details[model_name] = detail

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_overall_params, f, indent=4)

    details_path = out_path.with_name(f"{out_path.stem}_selection.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(best_selection_details, f, indent=4)

    logger.info(f"\nGrid Search Complete! Best parameters saved to '{out_path}'.")
    logger.info(f"Selection details saved to '{details_path}'.")


if __name__ == "__main__":
    main()
