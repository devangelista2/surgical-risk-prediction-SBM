from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "configs"
DATA_ROOT = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_ROOT = BASE_DIR / "outputs"
STUDIO_RUNS_ROOT = OUTPUTS_ROOT / "studio_runs"
TUNING_CACHE_ROOT = OUTPUTS_ROOT / "tuning_cache"
FREEZES_ROOT = OUTPUTS_ROOT / "freezes"
SEARCH_SPACE_PATH = CONFIG_DIR / "grid_search.json"
DATASET_SUFFIXES = {".xlsx", ".xls", ".csv"}
EXPORT_DPI = 150
TRAINING_IMPORT_CHECK = "import joblib, numpy, openpyxl, pandas, sklearn, tqdm"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

plt.rcParams.update({
    "figure.dpi": EXPORT_DPI,
    "savefig.dpi": EXPORT_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

app = Flask(__name__)
app.secret_key = "sbm-stratify-2024"
LAUNCH_JOBS: dict[str, dict[str, Any]] = {}
LAUNCH_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_json_loose(path: str | Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except (json.JSONDecodeError, OSError):
        return {}


def _python_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_override = os.environ.get("TRAIN_PYTHON", "").strip()
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend(
        [
            Path(sys.executable),
            BASE_DIR / ".venv" / "Scripts" / "python.exe",
            BASE_DIR / ".venv" / "bin" / "python",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_training_python() -> tuple[str | None, str | None]:
    failures: list[str] = []
    for candidate in _python_candidates():
        if not candidate.exists():
            continue
        try:
            probe = subprocess.run(
                [str(candidate), "-c", TRAINING_IMPORT_CHECK],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        if probe.returncode == 0:
            return str(candidate), None

        details = (probe.stderr or probe.stdout or "").strip()
        if details:
            failures.append(f"{candidate}: {details.splitlines()[-1]}")
        else:
            failures.append(f"{candidate}: dependency check failed with exit code {probe.returncode}")

    if failures:
        return None, "No usable Python interpreter found for training. " + " | ".join(failures)
    return None, "No usable Python interpreter found for training."


def get_launch_job(job_id: str) -> dict[str, Any] | None:
    with LAUNCH_JOBS_LOCK:
        job = LAUNCH_JOBS.get(job_id)
        return copy.deepcopy(job) if job else None


def update_launch_job(job_id: str, **updates: Any) -> None:
    with LAUNCH_JOBS_LOCK:
        job = LAUNCH_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = utc_now_iso()


def update_launch_execution_row(
    execution_rows: list[dict[str, Any]], target: str, **updates: Any
) -> list[dict[str, Any]]:
    for row in execution_rows:
        if row.get("target") == target:
            row.update(updates)
            break
    return execution_rows


def target_folder(target: str) -> str:
    """Folder name for a target; characters a path can't hold become '_', as in 'Complications (Y_N)'."""
    return re.sub(r'[\\/:*?"<>|]', "_", target)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def list_datasets() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir()
                  if p.is_file() and p.suffix.lower() in DATASET_SUFFIXES)


def dataset_path(name: str) -> Path:
    if name not in list_datasets():
        raise ValueError(f"Unknown dataset: {name}")
    return DATA_ROOT / name


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def file_digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def column_type(s: pd.Series) -> str:
    """Guess one of numeric, binary, categorical or date for a column."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if set(s.dropna().unique()) <= {0, 1} else "numeric"
    values = s.dropna().astype(str)
    if len(values) and pd.to_datetime(values, errors="coerce", format="mixed").notna().mean() > 0.9:
        return "date"
    return "categorical"


def dataset_profile(name: str) -> dict[str, Any]:
    df = read_table(dataset_path(name))
    columns = []
    for col, s in df.items():
        kind = column_type(s)
        columns.append({
            "name": str(col),
            "type": kind,
            "missing": int(s.isna().sum()),
            # A column unique on every row is an identifier, not a predictor.
            "id_like": kind in {"numeric", "categorical"} and s.nunique() == len(df),
        })
    return {"dataset": name, "rows": len(df), "columns": columns}


def default_search_space() -> dict[str, Any]:
    # ridge is regression-only and every studio target is binary.
    return {k: v for k, v in load_json(SEARCH_SPACE_PATH).items() if k != "ridge"}


def run_settings(run_root: Path) -> dict[str, Any]:
    """The launch settings of a run: the saved studio settings, else rebuilt from metadata.json."""
    saved = load_json_loose(run_root / "_runtime" / "studio_settings.json")
    if saved:
        return saved
    metas = [load_json(p) for p in sorted(run_root.glob("*/metadata.json"))]
    if not metas:
        raise ValueError("This run has no metadata.json.")
    first = metas[0]
    data = first.get("data_configuration", {})
    split = first.get("split_config", {})
    policy = first.get("binary_decision_policy", {})
    return {
        "dataset": Path(data.get("input_file", "")).name,
        "targets": [m.get("target_column") for m in metas],
        "models": sorted({m for meta in metas for m in meta.get("models_trained", [])}),
        "selected_features": data.get("input_features", []),
        "cols_string": data.get("cols_string", []),
        "cols_date": data.get("cols_date", []),
        "cols_multi": data.get("cols_multi", []),
        "split_strategy": first.get("split_strategy", "temporal"),
        "test_size": split.get("test_size") or 0.2,
        "split_column": split.get("split_column") or "Split",
        "date_column": split.get("date_column") or "Date of surgery",
        "threshold_val_size": policy.get("threshold_val_size") or 0.2,
        "min_recall": policy.get("min_recall") or 0.9,
        "f_beta": policy.get("f_beta") or 2.0,
        "fn_cost": policy.get("fn_cost") or 5.0,
        "fp_cost": policy.get("fp_cost") or 1.0,
    }


TUNING_KEY_FIELDS = (
    "dataset_digest", "selected_features", "cols_string", "cols_date", "cols_multi",
    "date_column", "test_size", "threshold_val_size", "min_recall", "f_beta", "fn_cost", "fp_cost",
)


def tuning_cache_path(settings: dict[str, Any], target: str, model: str, grid: Any) -> Path:
    """Cache file for one model's tuned parameters; any change to data, features, grid or policy gives a new file."""
    basis = {k: settings[k] for k in TUNING_KEY_FIELDS}
    basis.update(target=target, model=model, grid=grid)
    key = hashlib.sha1(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]
    return TUNING_CACHE_ROOT / f"{key}.json"


def build_training_plan(settings: dict[str, Any], search_space: dict[str, Any]) -> list[dict[str, Any]]:
    plan = []
    for target in settings["targets"]:
        cached, to_tune = {}, []
        for model in settings["models"]:
            if model not in search_space:
                continue
            path = tuning_cache_path(settings, target, model, search_space[model])
            hit = None if settings.get("force_retune") else load_json_loose(path)
            if hit:
                cached[model] = hit["params"]
            else:
                to_tune.append(model)
        plan.append({"target": target, "cached": cached, "to_tune": to_tune})
    return plan


FREEZE_FILES = ("pipeline.joblib", "decision_policy.json", "metrics.json")


def freeze_run(run_root: Path, picks: dict[str, str], high_pct: float, name: str) -> dict[str, Any]:
    """Copy one model per outcome into outputs/freezes/<name>/ with a bands.json, in the layout neurosurg-predict loads."""
    out = FREEZES_ROOT / slugify(name)
    if out.exists():
        raise ValueError(f"A freeze named '{out.name}' already exists; pick another name.")
    bands: dict[str, Any] = {
        "_comment": (f"'low' is the model's operating threshold; 'high' is the {high_pct:g}th "
                     "percentile of predicted probability in the test set."),
    }
    warnings = []
    for target, model in picks.items():
        src = run_root / target / model
        if src.resolve().parent.parent != run_root.resolve() or not (src / "pipeline.joblib").is_file():
            raise ValueError(f"{target} / {model} is not a trained model in this run.")
        policy = load_json(src / "decision_policy.json")
        if not policy or not (src / "test_predictions.csv").exists():
            raise ValueError(f"{target} / {model} has no test predictions; retrain it in this studio first.")
        probs = pd.read_csv(src / "test_predictions.csv")["y_prob"]
        bands[target] = {
            "low": round(float(policy["threshold"]), 3),
            "high": round(float(np.percentile(probs, high_pct)), 3),
        }
        data = load_json(run_root / target / "metadata.json").get("data_configuration", {})
        if model.startswith("torch") or data.get("cols_date") or data.get("cols_multi"):
            warnings.append(f"{target}: {model} needs this repo's src/ to load, so neurosurg-predict cannot load it.")
    for target, model in picks.items():
        (out / target / model).mkdir(parents=True)
        shutil.copy2(run_root / target / "metadata.json", out / target / "metadata.json")
        for f in FREEZE_FILES:
            shutil.copy2(run_root / target / model / f, out / target / model / f)
    save_json(out / "bands.json", bands)
    return {"name": out.name, "path": str(out), "bands": bands, "warnings": warnings}


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.08)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return "data:image/png;base64," + data


def fig_to_bytes(fig, fmt: str = "png") -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.08)
    buf.seek(0)
    return buf.read()


def primary_metric_name(task_type: str, summary_df: pd.DataFrame) -> str:
    candidates = {
        "binary": ["roc_auc", "average_precision", "recall", "accuracy"],
        "categorical": ["f1_macro", "accuracy"],
        "continuous": ["r2", "rmse", "mae"],
    }
    for metric in candidates.get(task_type, []):
        if metric in summary_df.columns:
            return metric
    numeric = [c for c in summary_df.columns
               if pd.api.types.is_numeric_dtype(summary_df[c]) and c not in {"fit_seconds"}]
    return numeric[0] if numeric else ""


def format_metric_value(metric_name: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if metric_name == "fit_seconds":
        return f"{float(value):.1f}s"
    return f"{float(value):.3f}"


def _dark_fig(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#0d1727")
    return fig, ax


def build_bar_chart(summary_df: pd.DataFrame, task_type: str) -> tuple[str | None, str]:
    metric = primary_metric_name(task_type, summary_df)
    if not metric or metric not in summary_df.columns:
        return None, ""
    df = (summary_df[summary_df["status"] == "ok"].copy()
          if "status" in summary_df.columns else summary_df.copy())
    if df.empty:
        return None, metric
    ascending = metric in {"rmse", "mae"}
    df = df.sort_values(metric, ascending=ascending)
    sns.set_theme(style="dark")
    fig, ax = _dark_fig(7.5, 4.2)
    palette = ["#5eead4", "#60a5fa", "#38bdf8", "#f59e0b", "#f472b6", "#c084fc"]
    sns.barplot(data=df, x=metric, y="model", hue="model", dodge=False,
                palette=palette[:len(df)], ax=ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title(f"Model comparison — {metric}", color="white", fontsize=13, pad=12)
    ax.set_xlabel(metric, color="#dbeafe")
    ax.set_ylabel("")
    ax.tick_params(colors="#dbeafe")
    for spine in ax.spines.values():
        spine.set_color("#1f314f")
    ax.grid(axis="x", color="#24364f", alpha=0.4)
    fig.tight_layout()
    return fig_to_b64(fig), metric


def build_heatmap_chart(summary_df: pd.DataFrame, task_type: str) -> str | None:
    candidates = {
        "binary": ["roc_auc", "average_precision", "recall", "precision",
                   "specificity", "f_beta", "accuracy"],
        "categorical": ["f1_macro", "accuracy", "roc_auc_ovr"],
        "continuous": ["r2", "rmse", "mae"],
    }
    metrics = [m for m in candidates.get(task_type, []) if m in summary_df.columns]
    if not metrics:
        return None
    df = (summary_df[summary_df["status"] == "ok"][["model", *metrics]].copy()
          if "status" in summary_df.columns else summary_df[["model", *metrics]].copy())
    if df.empty:
        return None
    df = df.set_index("model")
    fig, ax = _dark_fig(max(6, len(metrics) * 1.1), max(2.5, len(df) * 0.65))
    sns.heatmap(df, annot=True, fmt=".3f",
                cmap=sns.color_palette(["#0f172a", "#1d4ed8", "#2dd4bf"], as_cmap=True),
                linewidths=0.6, linecolor="#14233b", cbar=False, ax=ax)
    ax.set_title("Metric matrix", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#dbeafe", labelrotation=0)
    fig.tight_layout()
    return fig_to_b64(fig)


def aggregate_importances(target_dir: Path) -> pd.DataFrame:
    frames = []
    for model_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
        fi_path = model_dir / "feature_importance.csv"
        if not fi_path.exists():
            continue
        frame = pd.read_csv(fi_path)
        if frame.empty:
            continue
        frame["Model"] = model_dir.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("Feature", as_index=False)
        .agg(
            mean_importance=("Importance", "mean"),
            mean_abs_importance=("Importance", lambda s: float(np.mean(np.abs(s)))),
            std_importance=("Importance", "std"),
            models_reported=("Model", "nunique"),
        )
        .fillna({"std_importance": 0.0})
        .sort_values("mean_abs_importance", ascending=False)
    )


def build_importance_chart(aggregated: pd.DataFrame) -> str | None:
    if aggregated.empty:
        return None
    top = aggregated.head(15).sort_values("mean_abs_importance", ascending=True)
    fig, ax = _dark_fig(8, max(4.2, len(top) * 0.35))
    ax.barh(top["Feature"], top["mean_abs_importance"],
            color="#5eead4", alpha=0.85, edgecolor="#99f6e4")
    ax.set_title("Cross-model permutation importance", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Mean absolute importance", color="#dbeafe")
    ax.tick_params(colors="#dbeafe")
    for spine in ax.spines.values():
        spine.set_color("#1f314f")
    ax.grid(axis="x", color="#24364f", alpha=0.35)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_group_dist_chart(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    dist = pd.Series(counts).sort_values(ascending=True)
    fig, ax = _dark_fig(6.4, max(3.0, len(dist) * 0.55))
    ax.barh(dist.index, dist.values, color="#60a5fa", alpha=0.9)
    ax.set_title("Selected features by group", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#dbeafe")
    ax.set_xlabel("Count", color="#dbeafe")
    for spine in ax.spines.values():
        spine.set_color("#1f314f")
    ax.grid(axis="x", color="#24364f", alpha=0.35)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Run discovery and results
# ---------------------------------------------------------------------------

def discover_runs() -> list[dict[str, Any]]:
    groups: dict[Path, set[str]] = {}
    metadata_paths = [*OUTPUTS_ROOT.rglob("metadata.json"), *MODELS_DIR.rglob("metadata.json")]
    for metadata_path in metadata_paths:
        target_dir = metadata_path.parent
        summary_path = target_dir / "benchmark_summary.csv"
        if not summary_path.exists():
            continue
        run_root = target_dir.parent
        groups.setdefault(run_root, set()).add(target_dir.name)

    runs = []
    for run_root, targets in groups.items():
        if run_root in {OUTPUTS_ROOT, MODELS_DIR}:
            continue
        rel_path = run_root.relative_to(BASE_DIR)
        updated_at = datetime.fromtimestamp(run_root.stat().st_mtime)
        runs.append({
            "path": str(run_root),
            "rel_path": str(rel_path),
            "targets": sorted(targets),
            "label": f"{rel_path}  |  {len(targets)} target(s)  |  {updated_at.strftime('%Y-%m-%d %H:%M')}",
            "updated_at": updated_at.isoformat(),
        })
    return sorted(runs, key=lambda r: r["updated_at"], reverse=True)


def get_run_results(run_root: Path) -> dict[str, Any]:
    target_dirs = sorted(
        [c for c in run_root.iterdir()
         if c.is_dir() and c.name != "_runtime"
         and (c / "metadata.json").exists()
         and (c / "benchmark_summary.csv").exists()],
        key=lambda p: p.name,
    )

    results = []
    for target_dir in target_dirs:
        metadata = load_json(target_dir / "metadata.json")
        summary_path = target_dir / "benchmark_summary.csv"
        summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        task_type = metadata.get("task_type", "binary")

        ok_rows = (summary_df[summary_df["status"] == "ok"].copy()
                   if not summary_df.empty and "status" in summary_df.columns
                   else summary_df.copy())
        selected_metric = primary_metric_name(task_type, summary_df) if not summary_df.empty else ""

        cards: dict[str, Any] = {
            "target": target_dir.name,
            "split_strategy": metadata.get("split_strategy", "unknown").upper(),
            "feature_count": len(metadata.get("data_configuration", {}).get("input_features", [])),
            "model_count": len(ok_rows),
            "best_metric_name": None,
            "best_metric_value": None,
            "best_model": None,
        }
        if not ok_rows.empty and selected_metric in ok_rows.columns:
            ascending = selected_metric in {"rmse", "mae"}
            best_idx = (ok_rows[selected_metric].idxmin() if ascending
                        else ok_rows[selected_metric].idxmax())
            cards["best_metric_name"] = selected_metric
            cards["best_metric_value"] = format_metric_value(
                selected_metric, ok_rows.loc[best_idx, selected_metric])
            cards["best_model"] = str(ok_rows.loc[best_idx, "model"]).upper()

        preferred_cols = [c for c in [
            "model", "status", "roc_auc", "average_precision", "recall", "precision",
            "specificity", "f_beta", "accuracy", "r2", "rmse", "mae", "fit_seconds",
        ] if c in summary_df.columns]
        summary_table = (summary_df[preferred_cols].to_dict(orient="records")
                         if preferred_cols else [])

        bar_img, bar_metric = build_bar_chart(summary_df, task_type)
        heatmap_img = build_heatmap_chart(summary_df, task_type)

        aggregated = aggregate_importances(target_dir)
        importance_img = build_importance_chart(aggregated)
        weakest: list[dict] = []
        if not aggregated.empty:
            w = aggregated.sort_values("mean_abs_importance", ascending=True).head(10)
            weakest = w.rename(columns={
                "Feature": "Candidate to remove",
                "mean_importance": "Mean importance",
                "mean_abs_importance": "Mean abs importance",
                "models_reported": "Models",
            }).to_dict(orient="records")

        combined_curves = []
        for plot_name in ["combined_roc_curve.png", "combined_pr_curve.png"]:
            p = target_dir / plot_name
            if p.exists():
                combined_curves.append({
                    "path": str(p),
                    "title": plot_name.replace(".png", "").replace("_", " ").title(),
                })

        model_names = []
        if not summary_df.empty and "model" in summary_df.columns:
            model_names = summary_df["model"].tolist()
        else:
            model_names = sorted([c.name for c in target_dir.iterdir() if c.is_dir()])

        models_data = []
        for model_name in model_names:
            model_dir = target_dir / model_name
            metrics_raw = load_json(model_dir / "metrics.json")
            scalar_metrics = {
                k: format_metric_value(k, v)
                for k, v in metrics_raw.items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            }
            plot_files = [
                "roc_curve.png", "pr_curve.png", "confusion_matrix.png",
                "feature_importance.png", "actual_vs_predicted.png",
            ]
            available_plots = [
                {"path": str(model_dir / p),
                 "title": p.replace(".png", "").replace("_", " ").title()}
                for p in plot_files if (model_dir / p).exists()
            ]
            fi_path = model_dir / "feature_importance.csv"
            fi_table: list[dict] = []
            if fi_path.exists():
                fi_df = pd.read_csv(fi_path)
                fi_table = fi_df.head(20).to_dict(orient="records")
            models_data.append({
                "name": model_name,
                "metrics": scalar_metrics,
                "plots": available_plots,
                "fi_table": fi_table,
                "raw_metrics": metrics_raw,
            })

        results.append({
            "target": target_dir.name,
            "cards": cards,
            "summary_table": summary_table,
            "summary_columns": preferred_cols,
            "bar_img": bar_img,
            "bar_metric": bar_metric,
            "heatmap_img": heatmap_img,
            "importance_img": importance_img,
            "weakest": weakest,
            "combined_curves": combined_curves,
            "models": models_data,
        })

    return {"targets": results, "run_path": str(run_root)}


def run_subprocess(
    cmd: list[str], log_path: Path, progress_path: Path, on_progress
) -> tuple[int, str]:
    """Run one pipeline script, calling on_progress with its progress file until it exits."""
    if progress_path.exists():
        progress_path.unlink()
    child_env = os.environ.copy()
    child_env.setdefault("LOKY_MAX_CPU_COUNT", "1")
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("PYTHONUTF8", "1")
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        while proc.poll() is None:
            on_progress(load_json_loose(progress_path))
            time.sleep(0.8)
    return proc.returncode, log_path.read_text(encoding="utf-8", errors="replace").strip()


def policy_args(settings: dict[str, Any]) -> list[str]:
    return [
        "--min_recall", str(settings["min_recall"]),
        "--f_beta", str(settings["f_beta"]),
        "--fn_cost", str(settings["fn_cost"]),
        "--fp_cost", str(settings["fp_cost"]),
    ]


def run_training_job(
    job_id: str,
    settings: dict[str, Any],
    plan: list[dict[str, Any]],
    run_root: Path,
    training_python: str,
    total_units: int,
    execution_rows: list[dict[str, Any]],
) -> None:
    try:
        runtime_root = run_root / "_runtime"
        data_config_path = runtime_root / "data_config.json"
        search_space = settings["search_space"]
        completed_units = 0

        update_launch_job(job_id, status="running", started_at=utc_now_iso(),
                          current_step="Preparing training runtime...")

        def report(target: str, units: int, progress: dict[str, Any], fallback: str) -> None:
            step = progress.get("current_step") or fallback
            update_launch_execution_row(execution_rows, target, status="running", message=step)
            update_launch_job(
                job_id,
                execution=copy.deepcopy(execution_rows),
                current_target=target,
                current_model=progress.get("current_model"),
                current_step=step,
                completed_units=units,
                progress_pct=round(100.0 * units / total_units, 1) if total_units else 100.0,
            )

        for i, item in enumerate(plan):
            target = item["target"]
            slug = slugify(target)
            params = dict(item["cached"])
            planned = len(item["cached"]) + len(item["to_tune"])
            logs = []

            if item["to_tune"]:
                space_path = runtime_root / f"{slug}-search-space.json"
                save_json(space_path, {m: search_space[m] for m in item["to_tune"]})
                tuned_path = runtime_root / f"{slug}-tuned.json"
                test_size = float(settings["test_size"])
                cmd = [
                    training_python, str(SRC_DIR / "tune.py"),
                    "--target", target,
                    "--data_config", str(data_config_path),
                    "--search_space", str(space_path),
                    "--output_file", str(tuned_path),
                    "--date_column", settings["date_column"],
                    "--test_size", str(test_size),
                    # Same validation slice that train.py later uses to pick the threshold.
                    "--val_size", str(float(settings["threshold_val_size"]) * (1 - test_size)),
                    *policy_args(settings),
                    "--progress_path", str(runtime_root / f"{slug}-tune-progress.json"),
                ]
                base = completed_units
                rc, log = run_subprocess(
                    cmd, runtime_root / f"{slug}-tune.log", runtime_root / f"{slug}-tune-progress.json",
                    lambda p: report(target, base + int(p.get("completed_models", 0)), p, f"Tuning {target}..."),
                )
                logs.append(log)
                completed_units += len(item["to_tune"])
                tuned = load_json_loose(tuned_path)
                selection = load_json_loose(tuned_path.with_name(f"{tuned_path.stem}_selection.json"))
                for model in item["to_tune"]:
                    if isinstance(tuned.get(model), dict):
                        params[model] = tuned[model]
                        save_json(
                            tuning_cache_path(settings, target, model, search_space[model]),
                            {"params": tuned[model], "selection": selection.get(model)},
                        )

            models = [m for m in settings["models"] if isinstance(params.get(m), dict)]
            if not models:
                completed_units += planned
                update_launch_execution_row(
                    execution_rows, target, status="failed", returncode=None,
                    log="\n\n".join(logs)[-4000:],
                    message="No model could be tuned. See console log.",
                )
                update_launch_job(job_id, execution=copy.deepcopy(execution_rows),
                                  completed_units=completed_units, completed_targets=i + 1)
                continue

            model_config_path = runtime_root / f"{slug}-model-config.json"
            save_json(model_config_path, {m: params[m] for m in models})
            cmd = [
                training_python, str(SRC_DIR / "train.py"),
                "--target", target,
                "--data_config", str(data_config_path),
                "--model_config", str(model_config_path),
                "--models", ",".join(models),
                "--output_folder", str(run_root / target_folder(target)),
                "--split_strategy", settings["split_strategy"],
                "--test_size", str(settings["test_size"]),
                "--split_column", settings["split_column"],
                "--date_column", settings["date_column"],
                "--feature_importance",
                "--threshold_val_size", str(settings["threshold_val_size"]),
                *policy_args(settings),
                "--progress_path", str(runtime_root / f"{slug}-progress.json"),
            ]
            base = completed_units
            rc, log = run_subprocess(
                cmd, runtime_root / f"{slug}.log", runtime_root / f"{slug}-progress.json",
                lambda p: report(target, base + int(p.get("completed_models", 0)), p, f"Training {target}..."),
            )
            logs.append(log)
            completed_units += planned
            update_launch_execution_row(
                execution_rows, target,
                status="ok" if rc == 0 else "failed",
                models=", ".join(models),
                returncode=rc,
                log="\n\n".join(logs)[-4000:],
                message="" if rc == 0 else "Training subprocess failed. See console log.",
            )
            update_launch_job(
                job_id,
                execution=copy.deepcopy(execution_rows),
                completed_units=completed_units,
                completed_targets=i + 1,
                progress_pct=round(100.0 * completed_units / total_units, 1) if total_units else 100.0,
            )

        run_results = None
        try:
            run_results = get_run_results(run_root)
        except Exception:
            pass

        update_launch_job(
            job_id,
            status="completed",
            finished_at=utc_now_iso(),
            execution=copy.deepcopy(execution_rows),
            run_results=run_results,
            current_target=None,
            current_model=None,
            current_step="Training completed.",
            completed_units=total_units,
            completed_targets=len(plan),
            progress_pct=100.0,
        )
    except Exception as exc:
        update_launch_job(
            job_id,
            status="failed",
            finished_at=utc_now_iso(),
            error=str(exc),
            current_step="Training failed.",
        )


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "datasets": list_datasets(),
        "search_space": default_search_space(),
    })


@app.route("/api/datasets", methods=["POST"])
def upload_dataset():
    upload = request.files.get("file")
    name = secure_filename(upload.filename or "") if upload else ""
    if upload is None or Path(name).suffix.lower() not in DATASET_SUFFIXES:
        return jsonify({"error": "Upload an .xlsx, .xls or .csv file."}), 400
    DATA_ROOT.mkdir(exist_ok=True)
    upload.save(DATA_ROOT / name)
    return jsonify({"dataset": name, "datasets": list_datasets()})


@app.route("/api/dataset/profile")
def get_dataset_profile():
    try:
        return jsonify(dataset_profile(request.args.get("name", "")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/run/settings")
def get_run_settings():
    try:
        return jsonify(run_settings(Path(request.args.get("path", ""))))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/chart/group-distribution", methods=["POST"])
def group_distribution_chart():
    data = request.get_json()
    return jsonify({"image": build_group_dist_chart(data.get("counts", {}))})


@app.route("/api/launch", methods=["POST"])
def launch_training():
    data = request.get_json()
    settings: dict[str, Any] = {
        "dataset": data.get("dataset", ""),
        "targets": data.get("targets", []),
        "models": data.get("models", []),
        "selected_features": data.get("selected_features", []),
        "cols_string": data.get("cols_string", []),
        "cols_date": data.get("cols_date", []),
        "cols_multi": data.get("cols_multi", []),
        "run_name": data.get("run_name", ""),
        "split_strategy": data.get("split_strategy", "temporal"),
        "test_size": float(data.get("test_size", 0.20)),
        "threshold_val_size": float(data.get("threshold_val_size", 0.20)),
        "min_recall": float(data.get("min_recall", 0.90)),
        "f_beta": float(data.get("f_beta", 2.0)),
        "fn_cost": float(data.get("fn_cost", 5.0)),
        "fp_cost": float(data.get("fp_cost", 1.0)),
        "split_column": data.get("split_column", "Split"),
        "date_column": data.get("date_column", "Date of surgery"),
        "force_retune": bool(data.get("force_retune")),
        "search_space": data.get("search_space") or default_search_space(),
    }
    try:
        path = dataset_path(settings["dataset"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    settings["dataset_digest"] = file_digest(path)
    features = set(settings["selected_features"])
    for key in ("cols_string", "cols_date", "cols_multi"):
        settings[key] = [c for c in settings[key] if c in features]

    training_python, training_error = resolve_training_python()
    if not training_python:
        return jsonify({"error": training_error}), 500

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = STUDIO_RUNS_ROOT / f"{timestamp}-{slugify(settings['run_name'] or 'studio-run')}"
    runtime_root = run_root / "_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    save_json(runtime_root / "studio_settings.json", settings)
    save_json(runtime_root / "data_config.json", {
        "input_file": str(path),
        "input_features": settings["selected_features"],
        "cols_string": settings["cols_string"],
        "cols_date": settings["cols_date"],
        "cols_multi": settings["cols_multi"],
    })

    plan = build_training_plan(settings, settings["search_space"])
    total_units = sum(2 * len(i["to_tune"]) + len(i["cached"]) for i in plan)
    execution_rows = [
        {
            "target": item["target"],
            "status": "queued",
            "models": ", ".join(settings["models"]),
            "tuning": f"{len(item['cached'])} reused, {len(item['to_tune'])} to tune",
            "returncode": None,
            "log": "",
            "output_dir": str(run_root / target_folder(item["target"])),
            "message": "Queued...",
        }
        for item in plan
    ]

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "run_path": str(run_root),
        "run_results": None,
        "python_executable": training_python,
        "execution": execution_rows,
        "progress_pct": 0.0,
        "completed_units": 0,
        "total_units": total_units,
        "completed_targets": 0,
        "total_targets": len(plan),
        "current_target": None,
        "current_model": None,
        "current_step": "Queued for training...",
        "error": None,
    }
    with LAUNCH_JOBS_LOCK:
        LAUNCH_JOBS[job_id] = job

    threading.Thread(
        target=run_training_job,
        args=(job_id, settings, plan, run_root, training_python, total_units, copy.deepcopy(execution_rows)),
        daemon=True,
    ).start()
    return jsonify(get_launch_job(job_id)), 202


@app.route("/api/launch/status")
def launch_training_status():
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "No job id provided"}), 400
    job = get_launch_job(job_id)
    if not job:
        return jsonify({"error": "Training job not found"}), 404
    return jsonify(job)


@app.route("/api/runs")
def list_runs():
    return jsonify({"runs": discover_runs()})


@app.route("/api/run/results")
def run_results():
    run_path = request.args.get("path", "")
    if not run_path:
        return jsonify({"error": "No path provided"}), 400
    run_root = Path(run_path)
    if not run_root.exists():
        return jsonify({"error": "Run path does not exist"}), 404
    try:
        results = get_run_results(run_root)
        return jsonify(results)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/freeze", methods=["POST"])
def create_freeze():
    data = request.get_json()
    run_root = Path(data.get("run_path", ""))
    try:
        return jsonify(freeze_run(
            run_root, data.get("picks", {}), float(data.get("high_pct", 90)), data.get("name") or run_root.name,
        ))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/freeze/download")
def download_freeze():
    name = slugify(request.args.get("name", ""))
    root = FREEZES_ROOT / name
    if not root.is_dir():
        return jsonify({"error": "Freeze not found"}), 404
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(root))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{name}.zip")


@app.route("/api/image")
def serve_image():
    path = request.args.get("path", "")
    if not path:
        return "No path", 400
    p = Path(path)
    if not p.exists():
        return "Not found", 404
    try:
        p.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return "Forbidden", 403
    return send_file(str(p), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
