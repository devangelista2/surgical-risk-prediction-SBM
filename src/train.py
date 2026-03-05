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
from sklearn.inspection import permutation_importance
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


def balanced_weights_from_y(y_series):
    y_np = np.asarray(y_series)
    classes, counts = np.unique(y_np, return_counts=True)
    n_samples = len(y_np)
    n_classes = len(classes)
    return {
        cls: float(n_samples / (n_classes * count))
        for cls, count in zip(classes, counts, strict=False)
    }


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
        new_params["class_weight"] = {
            k.item() if hasattr(k, "item") else k: v for k, v in weights.items()
        }

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


def binary_confusion_metrics(y_true_bin, y_pred_bin, beta=2.0, fn_cost=5.0, fp_cost=1.0):
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
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
    expected_cost = fn_cost * fn + fp_cost * fp
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "npv": float(npv),
        "f_beta": float(f_beta),
        "expected_cost": float(expected_cost),
    }


def select_binary_threshold(
    y_true_bin, y_prob_pos, min_recall=0.9, beta=2.0, fn_cost=5.0, fp_cost=1.0
):
    thresholds = np.linspace(0.01, 0.99, 199)
    candidates = []
    for thr in thresholds:
        y_pred_bin = (y_prob_pos >= thr).astype(int)
        metric = binary_confusion_metrics(
            y_true_bin, y_pred_bin, beta=beta, fn_cost=fn_cost, fp_cost=fp_cost
        )
        metric["threshold"] = float(thr)
        candidates.append(metric)

    feasible = [c for c in candidates if c["recall"] >= min_recall]
    if feasible:
        best = max(
            feasible,
            key=lambda c: (c["precision"], c["f_beta"], -c["expected_cost"]),
        )
        best["meets_recall_constraint"] = True
        return best

    best = max(
        candidates,
        key=lambda c: (c["recall"], c["precision"], c["f_beta"], -c["expected_cost"]),
    )
    best["meets_recall_constraint"] = False
    return best


def evaluate_and_save(
    model_name: str,
    pipeline: Pipeline,
    X_test,
    y_test,
    task_type: str,
    out_dir: Path,
    feature_importance: bool = False,
    decision_threshold: float | None = None,
    threshold_selection: dict | None = None,
    f_beta: float = 2.0,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
):
    metrics = {}
    plot_data = {}
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
        classes = getattr(pipeline.named_steps["model"], "classes_", np.unique(y_test))

        if task_type == "binary" and y_prob is not None:
            pos_idx = 1 if y_prob.shape[1] > 1 else 0
            neg_idx = 1 - pos_idx if len(classes) > 1 else 0
            y_prob_pos = y_prob[:, pos_idx]
            threshold = 0.5 if decision_threshold is None else float(decision_threshold)

            pos_label = classes[pos_idx]
            neg_label = classes[neg_idx]
            y_pred_bin = (y_prob_pos >= threshold).astype(int)
            y_pred = np.where(y_pred_bin == 1, pos_label, neg_label)

            y_true_bin = (y_test == pos_label).astype(int).to_numpy()

            metrics["selected_threshold"] = float(threshold)
            metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob_pos))
            metrics["average_precision"] = float(
                average_precision_score(y_true_bin, y_prob_pos)
            )
            metrics["log_loss"] = float(log_loss(y_true_bin, y_prob_pos))
            metrics.update(
                binary_confusion_metrics(
                    y_true_bin,
                    y_pred_bin,
                    beta=f_beta,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                )
            )

            if threshold_selection:
                metrics["threshold_selection"] = threshold_selection

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

            plot_data["y_true_bin"] = y_true_bin
            plot_data["y_prob_pos"] = y_prob_pos

            policy = {
                "threshold": float(threshold),
                "f_beta": float(f_beta),
                "fn_cost": float(fn_cost),
                "fp_cost": float(fp_cost),
            }
            if threshold_selection:
                policy["selection_on_validation"] = threshold_selection
            with open(model_dir / "decision_policy.json", "w", encoding="utf-8") as f:
                json.dump(policy, f, indent=4)

        elif task_type == "categorical" and y_prob is not None:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
            metrics["log_loss"] = float(log_loss(y_test, y_prob))
            metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        metrics["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        vis.plot_confusion_matrix(
            y_test,
            y_pred,
            classes,
            f"Confusion Matrix - {model_name.upper()}",
            model_dir / "confusion_matrix.png",
        )

    joblib.dump(pipeline, model_dir / "pipeline.joblib")
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    if feature_importance:
        try:
            scoring = "accuracy"
            if (
                task_type == "binary"
                and hasattr(pipeline, "predict_proba")
                and model_name not in {"torch_mlp", "torch_ft_transformer"}
            ):
                scoring = "roc_auc"
            elif task_type == "continuous":
                scoring = "r2"

            pi_results = permutation_importance(
                pipeline, X_test, y_test, scoring=scoring, n_repeats=5, random_state=42
            )
            features = X_test.columns.tolist()
            importances_mean = pi_results.importances_mean
            importances_std = pi_results.importances_std

            sorted_idx = importances_mean.argsort()
            sorted_features = [features[i] for i in sorted_idx]
            sorted_importances = importances_mean[sorted_idx]
            sorted_std = importances_std[sorted_idx]

            fi_df = pd.DataFrame(
                {
                    "Feature": sorted_features[::-1],
                    "Importance": sorted_importances[::-1],
                    "Std": sorted_std[::-1],
                }
            )
            fi_df.to_csv(model_dir / "feature_importance.csv", index=False)

            vis.plot_feature_importance(
                sorted_features,
                sorted_importances,
                sorted_std,
                f"Feature Importance ({model_name.upper()})",
                model_dir / "feature_importance.png",
            )
        except Exception as e:
            from utils.logger import logger

            logger.warning(
                f"Could not compute feature importance for {model_name}: {e}"
            )

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
    parser.add_argument(
        "--feature_importance",
        action="store_true",
        help="Whether to compute and visualize permutation feature importance.",
    )
    parser.add_argument(
        "--threshold_val_size",
        type=float,
        default=0.2,
        help="Validation fraction carved from train set to choose binary threshold.",
    )
    parser.add_argument(
        "--min_recall",
        type=float,
        default=0.9,
        help="Minimum target recall for binary threshold selection.",
    )
    parser.add_argument(
        "--f_beta",
        type=float,
        default=2.0,
        help="Beta used in F-beta for binary operating-point selection.",
    )
    parser.add_argument(
        "--fn_cost",
        type=float,
        default=5.0,
        help="Relative cost of false negatives for threshold selection.",
    )
    parser.add_argument(
        "--fp_cost",
        type=float,
        default=1.0,
        help="Relative cost of false positives for threshold selection.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_json(args.data_config)
    models_config = load_json(args.model_config)

    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
        models_to_train = {
            k: v
            for k, v in models_config.items()
            if k in selected_models and isinstance(v, dict)
        }
    else:
        models_to_train = {k: v for k, v in models_config.items() if isinstance(v, dict)}

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

    X_fit, y_fit = X_train, y_train
    X_val_threshold, y_val_threshold = None, None

    if task_type == "binary" and args.threshold_val_size > 0:
        can_split = len(X_train) > 20 and y_train.nunique() > 1
        if can_split and args.split_strategy == "temporal":
            val_n = int(len(X_train) * args.threshold_val_size)
            if 0 < val_n < len(X_train):
                X_fit = X_train.iloc[:-val_n]
                y_fit = y_train.iloc[:-val_n]
                X_val_threshold = X_train.iloc[-val_n:]
                y_val_threshold = y_train.iloc[-val_n:]
        elif can_split:
            try:
                X_fit, X_val_threshold, y_fit, y_val_threshold = train_test_split(
                    X_train,
                    y_train,
                    test_size=args.threshold_val_size,
                    random_state=42,
                    stratify=y_train,
                )
            except Exception:
                X_fit, y_fit = X_train, y_train

    formatted_cols = "\n".join(f"  - {col}" for col in df.columns.tolist())
    logger.info(f"Pre-processing done. Found columns:\n{formatted_cols}")

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
            "fit_samples": len(X_fit),
            "threshold_validation_samples": (
                len(X_val_threshold) if X_val_threshold is not None else 0
            ),
        },
        "models_trained": list(models_to_train.keys()),
        "data_configuration": data_config,
        "model_hyperparameters": models_to_train,
        "binary_decision_policy": {
            "threshold_val_size": (
                args.threshold_val_size if task_type == "binary" else None
            ),
            "min_recall": args.min_recall if task_type == "binary" else None,
            "f_beta": args.f_beta if task_type == "binary" else None,
            "fn_cost": args.fn_cost if task_type == "binary" else None,
            "fp_cost": args.fp_cost if task_type == "binary" else None,
        },
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
            tuned_params = apply_imbalance_strategy(model_name, task_type, params, y_fit)

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

            model = get_model(model_name, task_type, tuned_params)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

            if model_name in {"torch_mlp", "torch_ft_transformer"}:
                X_train_t = pipeline.named_steps["preprocess"].fit_transform(
                    X_fit, y_fit
                )
                if X_val_threshold is not None:
                    X_eval_t = pipeline.named_steps["preprocess"].transform(
                        X_val_threshold
                    )
                    pipeline.named_steps["model"].fit(
                        X_train_t, y_fit, eval_set=(X_eval_t, y_val_threshold)
                    )
                else:
                    pipeline.named_steps["model"].fit(X_train_t, y_fit)
            else:
                fit_kwargs = {}
                if model_name == "hgb" and task_type in {"binary", "categorical"}:
                    fit_kwargs["model__sample_weight"] = sample_weight_from_y(y_fit)
                pipeline.fit(X_fit, y_fit, **fit_kwargs)

            fit_seconds = round(perf_counter() - start_time, 3)

            selected_threshold = None
            threshold_selection = None
            if (
                task_type == "binary"
                and X_val_threshold is not None
                and hasattr(pipeline, "predict_proba")
            ):
                y_val_prob = pipeline.predict_proba(X_val_threshold)
                pos_idx = 1 if y_val_prob.shape[1] > 1 else 0
                classes = getattr(
                    pipeline.named_steps["model"], "classes_", np.unique(y_val_threshold)
                )
                pos_label = classes[pos_idx]
                y_val_bin = (y_val_threshold == pos_label).astype(int).to_numpy()
                threshold_selection = select_binary_threshold(
                    y_val_bin,
                    y_val_prob[:, pos_idx],
                    min_recall=args.min_recall,
                    beta=args.f_beta,
                    fn_cost=args.fn_cost,
                    fp_cost=args.fp_cost,
                )
                selected_threshold = threshold_selection["threshold"]
                logger.info(
                    "Model %s selected threshold=%.3f (val recall=%.3f, precision=%.3f, meets_recall=%s)",
                    model_name,
                    selected_threshold,
                    threshold_selection["recall"],
                    threshold_selection["precision"],
                    threshold_selection["meets_recall_constraint"],
                )

            metrics, plot_data = evaluate_and_save(
                model_name,
                pipeline,
                X_test,
                y_test,
                task_type,
                out_dir,
                args.feature_importance,
                decision_threshold=selected_threshold,
                threshold_selection=threshold_selection,
                f_beta=args.f_beta,
                fn_cost=args.fn_cost,
                fp_cost=args.fp_cost,
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
