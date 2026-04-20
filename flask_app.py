from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import base64
import threading
import time
import uuid
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request, send_file, Response

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "configs"
GRIDSEARCH_ROOT = BASE_DIR / "gridsearch" / "preoperative"
OUTPUTS_ROOT = BASE_DIR / "outputs"
STUDIO_RUNS_ROOT = OUTPUTS_ROOT / "studio_runs"
PREOPERATIVE_CONFIG_PATH = CONFIG_DIR / "preoperative.json"
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


def build_training_plan(targets: list[str], models: list[str]) -> tuple[list[dict[str, Any]], int]:
    plan: list[dict[str, Any]] = []
    total_units = 0
    for target in targets:
        best_params = best_parameters_for_target(target)
        target_models = [m for m in models if isinstance(best_params.get(m), dict)]
        plan.append(
            {
                "target": target,
                "models": target_models,
                "model_params": {m: best_params[m] for m in target_models},
            }
        )
        total_units += len(target_models)
    return plan, total_units


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


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def feature_group(feature: str) -> str:
    if feature.startswith("Comorbidities_"):
        return "Comorbidities"
    if feature.startswith("Symptoms_"):
        return "Presentation"
    if feature.startswith("Radio_") or feature.startswith("Optic") or feature.startswith("ICA/"):
        return "Imaging & Anatomy"
    if "KPS" in feature or feature in {"ASA", "Charlson Comorbidity Index"}:
        return "Functional Status"
    if feature in {"Age", "Sex", "Date of surgery", "Date of Birth"}:
        return "Demographics & Timing"
    return "Clinical Core"


def available_targets() -> list[str]:
    if not GRIDSEARCH_ROOT.exists():
        return []
    return sorted([
        child.name for child in GRIDSEARCH_ROOT.iterdir()
        if child.is_dir() and (child / "best_parameters.json").exists()
    ])


def best_parameters_for_target(target: str) -> dict[str, Any]:
    return load_json(GRIDSEARCH_ROOT / target / "best_parameters.json")


def available_models_for_target(target: str) -> list[str]:
    params = best_parameters_for_target(target)
    return sorted([k for k, v in params.items() if isinstance(v, dict)])


def model_union_for_targets(targets: list[str]) -> list[str]:
    union: set[str] = set()
    for t in targets:
        union.update(available_models_for_target(t))
    return sorted(union)


def apply_preset(preset_name: str, all_features: list[str]) -> list[str]:
    if preset_name == "all":
        return list(all_features)
    if preset_name == "clear":
        return []
    if preset_name == "clinical":
        keep = {"Demographics & Timing", "Comorbidities", "Presentation", "Functional Status", "Clinical Core"}
        return [f for f in all_features if feature_group(f) in keep]
    if preset_name == "imaging":
        keep = {"Imaging & Anatomy", "Functional Status"}
        return [f for f in all_features if feature_group(f) in keep]
    if preset_name == "compact":
        compact = {
            "Age", "Sex", "Pre-Op KPS", "ASA", "Charlson Comorbidity Index",
            "Radio_Pre-Op max_axial_diam_mm", "Radio_Tumor Location", "Radio_Tumor side", "Radio_Edema",
        }
        return [f for f in all_features if f in compact]
    return list(all_features)


def build_subset_config(data_config: dict[str, Any], selected_features: list[str]) -> dict[str, Any]:
    selected_set = set(selected_features)
    ordered = [f for f in data_config.get("input_features", []) if f in selected_set]
    return {
        "input_file": data_config.get("input_file"),
        "input_features": ordered,
        "cols_string": [f for f in data_config.get("cols_string", []) if f in selected_set],
        "cols_date": [f for f in data_config.get("cols_date", []) if f in selected_set],
        "cols_multi": [f for f in data_config.get("cols_multi", []) if f in selected_set],
    }


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


def build_group_dist_chart(features: list[str]) -> str | None:
    if not features:
        return None
    dist = (
        pd.Series([feature_group(f) for f in features])
        .value_counts()
        .sort_values(ascending=True)
    )
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
    for metadata_path in OUTPUTS_ROOT.rglob("metadata.json"):
        target_dir = metadata_path.parent
        summary_path = target_dir / "benchmark_summary.csv"
        if not summary_path.exists():
            continue
        run_root = target_dir.parent
        groups.setdefault(run_root, set()).add(target_dir.name)

    runs = []
    for run_root, targets in groups.items():
        if run_root == OUTPUTS_ROOT:
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


def run_training_job(
    job_id: str,
    payload: dict[str, Any],
    training_plan: list[dict[str, Any]],
) -> None:
    try:
        run_root = Path(payload["run_root"])
        runtime_root = Path(payload["runtime_root"])
        data_config_path = Path(payload["data_config_path"])
        training_python = payload["training_python"]
        total_units = int(payload["total_units"])
        completed_units = 0
        completed_targets = 0
        execution_rows = copy.deepcopy(payload["execution"])

        update_launch_job(
            job_id,
            status="running",
            started_at=utc_now_iso(),
            current_step="Preparing training runtime...",
        )

        for item in training_plan:
            target = item["target"]
            target_models = item["models"]
            target_output_dir = run_root / target

            if not target_models:
                execution_rows = update_launch_execution_row(
                    execution_rows,
                    target,
                    status="skipped",
                    message="No saved best parameters for the selected models.",
                )
                completed_targets += 1
                update_launch_job(
                    job_id,
                    execution=copy.deepcopy(execution_rows),
                    completed_targets=completed_targets,
                )
                continue

            model_config_path = runtime_root / f"{slugify(target)}-model-config.json"
            save_json(model_config_path, item["model_params"])
            log_path = runtime_root / f"{slugify(target)}.log"
            progress_path = runtime_root / f"{slugify(target)}-progress.json"
            if progress_path.exists():
                progress_path.unlink()

            execution_rows = update_launch_execution_row(
                execution_rows,
                target,
                status="running",
                message="Starting training...",
            )
            update_launch_job(
                job_id,
                execution=copy.deepcopy(execution_rows),
                current_target=target,
                current_model=None,
                current_step=f"Preparing {target}...",
            )

            cmd = [
                training_python,
                str(SRC_DIR / "train.py"),
                "--target",
                target,
                "--data_config",
                str(data_config_path),
                "--model_config",
                str(model_config_path),
                "--models",
                ",".join(target_models),
                "--output_folder",
                str(target_output_dir),
                "--split_strategy",
                payload["split_strategy"],
                "--test_size",
                str(payload["test_size"]),
                "--split_column",
                payload["split_column"],
                "--date_column",
                payload["date_column"],
                "--feature_importance",
                "--threshold_val_size",
                str(payload["threshold_val_size"]),
                "--min_recall",
                str(payload["min_recall"]),
                "--f_beta",
                str(payload["f_beta"]),
                "--fn_cost",
                str(payload["fn_cost"]),
                "--fp_cost",
                str(payload["fp_cost"]),
                "--progress_path",
                str(progress_path),
            ]
            child_env = os.environ.copy()
            child_env.setdefault("LOKY_MAX_CPU_COUNT", "1")
            child_env.setdefault("PYTHONUNBUFFERED", "1")

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

                while True:
                    rc = proc.poll()
                    progress = load_json_loose(progress_path)
                    target_completed_models = int(progress.get("completed_models", 0))
                    current_model = progress.get("current_model")
                    current_step = progress.get("current_step") or f"Training {target}..."
                    message = progress.get("message") or current_step
                    progress_pct = (
                        round(100.0 * (completed_units + target_completed_models) / total_units, 1)
                        if total_units
                        else 100.0
                    )

                    execution_rows = update_launch_execution_row(
                        execution_rows,
                        target,
                        status="running",
                        message=message,
                    )
                    update_launch_job(
                        job_id,
                        execution=copy.deepcopy(execution_rows),
                        current_target=target,
                        current_model=current_model,
                        current_step=current_step,
                        completed_units=completed_units + target_completed_models,
                        progress_pct=progress_pct,
                    )

                    if rc is not None:
                        break
                    time.sleep(0.8)

            combined_log = log_path.read_text(encoding="utf-8", errors="replace").strip()
            completed_units += len(target_models)
            completed_targets += 1
            status = "ok" if proc.returncode == 0 else "failed"
            execution_rows = update_launch_execution_row(
                execution_rows,
                target,
                status=status,
                returncode=proc.returncode,
                log=combined_log[:4000],
                message="" if proc.returncode == 0 else "Training subprocess failed. See console log.",
            )
            update_launch_job(
                job_id,
                execution=copy.deepcopy(execution_rows),
                completed_units=completed_units,
                completed_targets=completed_targets,
                progress_pct=(round(100.0 * completed_units / total_units, 1) if total_units else 100.0),
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
            completed_targets=len(training_plan),
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
    data_config = load_json(PREOPERATIVE_CONFIG_PATH)
    all_features: list[str] = data_config.get("input_features", [])
    targets = available_targets()
    all_models = model_union_for_targets(targets)
    groups = sorted({feature_group(f) for f in all_features})
    group_map = {g: [f for f in all_features if feature_group(f) == g] for g in groups}
    return jsonify({
        "all_features": all_features,
        "targets": targets,
        "all_models": all_models,
        "groups": groups,
        "group_map": group_map,
    })


@app.route("/api/features/preset", methods=["POST"])
def feature_preset():
    data = request.get_json()
    preset = data.get("preset", "all")
    all_features: list[str] = data.get("all_features", [])
    return jsonify({"features": apply_preset(preset, all_features)})


@app.route("/api/features/group-action", methods=["POST"])
def feature_group_action():
    data = request.get_json()
    action = data.get("action", "add")
    group_name = data.get("group", "")
    all_features: list[str] = data.get("all_features", [])
    selected: list[str] = data.get("selected", [])
    group_features = [f for f in all_features if feature_group(f) == group_name]

    if action == "add":
        seen: set[str] = set(selected)
        new = list(selected)
        for f in group_features:
            if f not in seen:
                new.append(f)
    elif action == "remove":
        gset = set(group_features)
        new = [f for f in selected if f not in gset]
    elif action == "only":
        new = list(group_features)
    else:
        new = list(selected)

    return jsonify({"features": new})


@app.route("/api/chart/group-distribution", methods=["POST"])
def group_distribution_chart():
    data = request.get_json()
    features: list[str] = data.get("features", [])
    img = build_group_dist_chart(features)
    return jsonify({"image": img})


@app.route("/api/models-for-targets", methods=["POST"])
def models_for_targets():
    data = request.get_json()
    targets: list[str] = data.get("targets", [])
    models = model_union_for_targets(targets)
    availability = [
        {
            "target": t,
            "available_models": ", ".join(available_models_for_target(t)) or "none",
            "n_models": len(available_models_for_target(t)),
        }
        for t in targets
    ]
    return jsonify({"models": models, "availability": availability})


@app.route("/api/launch", methods=["POST"])
def launch_training():
    data = request.get_json()
    targets: list[str] = data.get("targets", [])
    models: list[str] = data.get("models", [])
    selected_features: list[str] = data.get("selected_features", [])
    run_name: str = data.get("run_name", "")
    split_strategy: str = data.get("split_strategy", "temporal")
    test_size = float(data.get("test_size", 0.20))
    threshold_val_size = float(data.get("threshold_val_size", 0.20))
    min_recall = float(data.get("min_recall", 0.90))
    f_beta = float(data.get("f_beta", 2.0))
    fn_cost = float(data.get("fn_cost", 5.0))
    fp_cost = float(data.get("fp_cost", 1.0))
    split_column: str = data.get("split_column", "Split")
    date_column: str = data.get("date_column", "Date of surgery")
    training_python, training_error = resolve_training_python()
    if not training_python:
        return jsonify({
            "execution": [],
            "run_path": "",
            "run_results": None,
            "error": training_error,
        }), 500

    base_config = load_json(PREOPERATIVE_CONFIG_PATH)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = slugify(run_name or "studio-run")
    run_root = STUDIO_RUNS_ROOT / f"{timestamp}-{label}"
    runtime_root = run_root / "_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    subset_config = build_subset_config(base_config, selected_features)
    data_config_path = runtime_root / "data_config.json"
    save_json(data_config_path, subset_config)
    training_plan, total_units = build_training_plan(targets, models)
    execution_rows: list[dict[str, Any]] = []
    for item in training_plan:
        target_models = item["models"]
        execution_rows.append(
            {
                "target": item["target"],
                "status": "queued" if target_models else "skipped",
                "models": ", ".join(target_models),
                "returncode": None,
                "log": "",
                "output_dir": str(run_root / item["target"]),
                "message": (
                    "Queued for training..."
                    if target_models
                    else "No saved best parameters for the selected models."
                ),
            }
        )

    job_id = uuid.uuid4().hex
    job_payload = {
        "run_root": str(run_root),
        "runtime_root": str(runtime_root),
        "data_config_path": str(data_config_path),
        "training_python": training_python,
        "split_strategy": split_strategy,
        "test_size": test_size,
        "threshold_val_size": threshold_val_size,
        "min_recall": min_recall,
        "f_beta": f_beta,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "split_column": split_column,
        "date_column": date_column,
        "total_units": total_units,
        "execution": copy.deepcopy(execution_rows),
    }
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
        "execution": copy.deepcopy(execution_rows),
        "progress_pct": 0.0,
        "completed_units": 0,
        "total_units": total_units,
        "completed_targets": 0,
        "total_targets": len(training_plan),
        "current_target": None,
        "current_model": None,
        "current_step": "Queued for training...",
        "error": None,
    }
    with LAUNCH_JOBS_LOCK:
        LAUNCH_JOBS[job_id] = job

    thread = threading.Thread(
        target=run_training_job,
        args=(job_id, job_payload, training_plan),
        daemon=True,
    )
    thread.start()
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
