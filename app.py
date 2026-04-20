from __future__ import annotations

import json
import re
import subprocess
import sys
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "configs"
GRIDSEARCH_ROOT = BASE_DIR / "gridsearch" / "preoperative"
OUTPUTS_ROOT = BASE_DIR / "outputs"
STUDIO_RUNS_ROOT = OUTPUTS_ROOT / "studio_runs"
PREOPERATIVE_CONFIG_PATH = CONFIG_DIR / "preoperative.json"
EXPORT_DPI = 300

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

plt.rcParams.update(
    {
        "figure.dpi": EXPORT_DPI,
        "savefig.dpi": EXPORT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


st.set_page_config(
    page_title="MedModel Studio",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CSS = """
<style>
    :root {
        --bg: #07111f;
        --panel: rgba(10, 20, 38, 0.72);
        --panel-strong: rgba(16, 29, 52, 0.92);
        --panel-soft: rgba(19, 37, 66, 0.62);
        --line: rgba(148, 163, 184, 0.18);
        --text: #e5eefc;
        --muted: #97aac8;
        --accent: #5eead4;
        --accent-2: #60a5fa;
        --accent-3: #f59e0b;
        --danger: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(96, 165, 250, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(94, 234, 212, 0.12), transparent 30%),
            linear-gradient(180deg, #08111f 0%, #07101a 55%, #050b13 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    label,
    .stAlert {
        color: var(--text);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.8rem 1.8rem 1.6rem 1.8rem;
        border-radius: 28px;
        border: 1px solid var(--line);
        background:
            linear-gradient(135deg, rgba(16, 29, 52, 0.96), rgba(8, 16, 31, 0.82)),
            linear-gradient(45deg, rgba(96, 165, 250, 0.25), rgba(94, 234, 212, 0.14));
        box-shadow: 0 28px 60px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.25rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        inset: auto -8% -35% auto;
        width: 340px;
        height: 340px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(94, 234, 212, 0.18), transparent 68%);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(94, 234, 212, 0.1);
        border: 1px solid rgba(94, 234, 212, 0.18);
        color: #bff8ef;
        font-size: 0.77rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero h1 {
        font-family: "Space Grotesk", "Aptos Display", "Segoe UI", sans-serif;
        font-size: 3rem;
        line-height: 1.02;
        letter-spacing: -0.05em;
        margin: 0.95rem 0 0.55rem 0;
        color: white;
    }

    .hero p {
        max-width: 72ch;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
    }

    .micro-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.25rem;
    }

    .micro-card {
        padding: 0.95rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(12, 22, 41, 0.68);
        backdrop-filter: blur(16px);
    }

    .micro-card .label {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .micro-card .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
    }

    .section-title {
        font-family: "Space Grotesk", "Aptos Display", "Segoe UI", sans-serif;
        font-size: 1.22rem;
        letter-spacing: -0.03em;
        margin: 0.15rem 0 0.3rem 0;
        color: white;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .feature-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.15rem;
    }

    .feature-chip {
        padding: 0.32rem 0.62rem;
        border-radius: 999px;
        background: rgba(96, 165, 250, 0.11);
        border: 1px solid rgba(96, 165, 250, 0.18);
        color: #cde4ff;
        font-size: 0.82rem;
    }

    .panel-note {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(12, 22, 41, 0.62);
        color: var(--muted);
        font-size: 0.94rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        padding: 0.3rem;
        background: rgba(8, 16, 31, 0.66);
        border: 1px solid var(--line);
        border-radius: 999px;
        width: fit-content;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.6rem;
        border-radius: 999px;
        background: transparent;
        color: var(--muted);
        padding: 0 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.18), rgba(94, 234, 212, 0.12));
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--line);
    }

    div[data-testid="stExpander"] details {
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(10, 20, 38, 0.56);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid rgba(96, 165, 250, 0.25);
        background: linear-gradient(135deg, rgba(20, 35, 62, 0.95), rgba(9, 18, 34, 0.95));
        color: white;
        min-height: 2.8rem;
        font-weight: 600;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.95), rgba(20, 184, 166, 0.85));
        color: #04101e;
        border-color: rgba(94, 234, 212, 0.3);
    }

    .stMultiSelect div[data-baseweb="select"],
    .stTextInput div[data-baseweb="input"],
    .stSelectbox div[data-baseweb="select"],
    .stNumberInput div[data-baseweb="input"] {
        background: rgba(12, 22, 41, 0.72);
        border-radius: 16px;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.2rem;
    }
</style>
"""


def inject_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


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


def feature_kind(feature: str, data_config: dict[str, Any]) -> str:
    if feature in data_config.get("cols_string", []):
        return "Categorical"
    if feature in data_config.get("cols_date", []):
        return "Date"
    if feature in data_config.get("cols_multi", []):
        return "Multi-label"
    return "Numeric / Binary"


@st.cache_data(show_spinner=False)
def load_base_feature_config() -> dict[str, Any]:
    return load_json(str(PREOPERATIVE_CONFIG_PATH))


def build_feature_catalog(data_config: dict[str, Any], selected: list[str]) -> pd.DataFrame:
    rows = []
    for feature in data_config.get("input_features", []):
        rows.append(
            {
                "use": feature in selected,
                "feature": feature,
                "group": feature_group(feature),
                "type": feature_kind(feature, data_config),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["group", "feature"]).reset_index(drop=True)


def build_subset_config(data_config: dict[str, Any], selected_features: list[str]) -> dict[str, Any]:
    selected_set = set(selected_features)
    ordered_features = [f for f in data_config.get("input_features", []) if f in selected_set]
    return {
        "input_file": data_config.get("input_file"),
        "input_features": ordered_features,
        "cols_string": [f for f in data_config.get("cols_string", []) if f in selected_set],
        "cols_date": [f for f in data_config.get("cols_date", []) if f in selected_set],
        "cols_multi": [f for f in data_config.get("cols_multi", []) if f in selected_set],
    }


def build_target_model_availability(targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        available = available_models_for_target(target)
        rows.append(
            {
                "target": target,
                "available_models": ", ".join(available) if available else "none",
                "n_models": len(available),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def available_targets() -> list[str]:
    if not GRIDSEARCH_ROOT.exists():
        return []
    targets = []
    for child in GRIDSEARCH_ROOT.iterdir():
        if child.is_dir() and (child / "best_parameters.json").exists():
            targets.append(child.name)
    return sorted(targets)


@st.cache_data(show_spinner=False)
def best_parameters_for_target(target: str) -> dict[str, Any]:
    return load_json(str(GRIDSEARCH_ROOT / target / "best_parameters.json"))


def available_models_for_target(target: str) -> list[str]:
    params = best_parameters_for_target(target)
    return sorted([name for name, value in params.items() if isinstance(value, dict)])


def model_union_for_targets(targets: list[str]) -> list[str]:
    union: set[str] = set()
    for target in targets:
        union.update(available_models_for_target(target))
    return sorted(union)


def metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="micro-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div style="color: var(--muted); font-size: 0.84rem; margin-top: 0.35rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def figure_download_bytes(fig, fmt: str) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format=fmt, dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.08)
    buffer.seek(0)
    return buffer.getvalue()


def render_figure_with_downloads(fig, base_name: str) -> None:
    st.pyplot(fig, use_container_width=True)
    download_cols = st.columns([1, 1, 3])
    png_bytes = figure_download_bytes(fig, "png")
    pdf_bytes = figure_download_bytes(fig, "pdf")
    with download_cols[0]:
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"{base_name}.png",
            mime="image/png",
            use_container_width=True,
        )
    with download_cols[1]:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{base_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def render_saved_plot(plot_path: Path, caption: str) -> None:
    st.image(str(plot_path), caption=caption, use_container_width=True)
    sibling_pdf = plot_path.with_suffix(".pdf")
    controls = st.columns([1, 1, 3])
    with controls[0]:
        st.download_button(
            "PNG",
            data=plot_path.read_bytes(),
            file_name=plot_path.name,
            mime="image/png",
            use_container_width=True,
            key=f"png-{plot_path}",
        )
    with controls[1]:
        if sibling_pdf.exists():
            st.download_button(
                "PDF",
                data=sibling_pdf.read_bytes(),
                file_name=sibling_pdf.name,
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf-{plot_path}",
            )


def build_filtered_feature_catalog(
    catalog_df: pd.DataFrame,
    search_query: str,
    selected_groups: list[str],
    selected_types: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if catalog_df.empty:
        return catalog_df, pd.Series(dtype=bool)

    mask = pd.Series(True, index=catalog_df.index)
    if search_query.strip():
        pattern = re.escape(search_query.strip())
        mask &= catalog_df["feature"].str.contains(pattern, case=False, regex=True)
    if selected_groups:
        mask &= catalog_df["group"].isin(selected_groups)
    if selected_types:
        mask &= catalog_df["type"].isin(selected_types)
    return catalog_df.loc[mask].copy(), mask


def render_feature_chips(features: list[str], limit: int = 28) -> None:
    if not features:
        st.markdown('<div class="panel-note">No features selected.</div>', unsafe_allow_html=True)
        return

    visible = features[:limit]
    chips = "".join(f'<span class="feature-chip">{feature}</span>' for feature in visible)
    more = ""
    if len(features) > limit:
        more = f'<span class="feature-chip">+{len(features) - limit} more</span>'
    st.markdown(
        f'<div class="feature-chip-wrap">{chips}{more}</div>',
        unsafe_allow_html=True,
    )


def set_selected_features(features: list[str], all_features: list[str]) -> None:
    ordered = [feature for feature in all_features if feature in set(features)]
    st.session_state.selected_features = ordered
    st.session_state.selected_features_picker = ordered


def init_feature_state(all_features: list[str]) -> None:
    if "selected_features" not in st.session_state:
        set_selected_features(list(all_features), all_features)
    else:
        prior = list(st.session_state.selected_features)
        current = [f for f in prior if f in all_features]
        if prior and not current:
            current = list(all_features)
        set_selected_features(current, all_features)


def apply_feature_preset(preset_name: str, all_features: list[str]) -> None:
    if preset_name == "all":
        set_selected_features(list(all_features), all_features)
    elif preset_name == "clear":
        set_selected_features([], all_features)
    elif preset_name == "clinical":
        set_selected_features(
            [
                feature
                for feature in all_features
                if feature_group(feature) in {
                    "Demographics & Timing",
                    "Comorbidities",
                    "Presentation",
                    "Functional Status",
                    "Clinical Core",
                }
            ],
            all_features,
        )
    elif preset_name == "imaging":
        set_selected_features(
            [
                feature
                for feature in all_features
                if feature_group(feature) in {"Imaging & Anatomy", "Functional Status"}
            ],
            all_features,
        )
    elif preset_name == "compact":
        compact = {
            "Age",
            "Sex",
            "Pre-Op KPS",
            "ASA",
            "Charlson Comorbidity Index",
            "Radio_Pre-Op max_axial_diam_mm",
            "Radio_Tumor Location",
            "Radio_Tumor side",
            "Radio_Edema",
        }
        set_selected_features(
            [feature for feature in all_features if feature in compact],
            all_features,
        )


def launch_training_run(
    *,
    targets: list[str],
    models: list[str],
    selected_features: list[str],
    base_config: dict[str, Any],
    run_name: str,
    split_strategy: str,
    test_size: float,
    split_column: str,
    date_column: str,
    threshold_val_size: float,
    min_recall: float,
    f_beta: float,
    fn_cost: float,
    fp_cost: float,
) -> tuple[Path, list[dict[str, Any]]]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = slugify(run_name or "studio-run")
    run_root = STUDIO_RUNS_ROOT / f"{timestamp}-{label}"
    runtime_root = run_root / "_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    subset_config = build_subset_config(base_config, selected_features)
    data_config_path = runtime_root / "data_config.json"
    save_json(data_config_path, subset_config)

    execution_rows: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Preparing training run...")
    status_box = st.empty()

    for index, target in enumerate(targets, start=1):
        best_params = best_parameters_for_target(target)
        target_models = [
            model_name
            for model_name in models
            if isinstance(best_params.get(model_name), dict)
        ]

        if not target_models:
            execution_rows.append(
                {
                    "target": target,
                    "status": "skipped",
                    "models": "",
                    "returncode": None,
                    "log_file": "",
                    "output_dir": str(run_root / target),
                    "message": "No saved best parameters for the selected models.",
                }
            )
            progress.progress(index / len(targets), text=f"Skipped {target}")
            continue

        model_config_path = runtime_root / f"{slugify(target)}-model-config.json"
        save_json(
            model_config_path,
            {model_name: best_params[model_name] for model_name in target_models},
        )

        target_output_dir = run_root / target
        log_path = runtime_root / f"{slugify(target)}.log"

        cmd = [
            sys.executable,
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
            split_strategy,
            "--test_size",
            str(test_size),
            "--split_column",
            split_column,
            "--date_column",
            date_column,
            "--feature_importance",
            "--threshold_val_size",
            str(threshold_val_size),
            "--min_recall",
            str(min_recall),
            "--f_beta",
            str(f_beta),
            "--fn_cost",
            str(fn_cost),
            "--fp_cost",
            str(fp_cost),
        ]

        status_box.info(f"Training `{target}` with {len(target_models)} model(s)...")
        completed = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        combined_log = (completed.stdout or "") + "\n" + (completed.stderr or "")
        log_path.write_text(combined_log.strip(), encoding="utf-8")

        execution_rows.append(
            {
                "target": target,
                "status": "ok" if completed.returncode == 0 else "failed",
                "models": ", ".join(target_models),
                "returncode": completed.returncode,
                "log_file": str(log_path),
                "output_dir": str(target_output_dir),
                "message": "",
            }
        )
        progress.progress(index / len(targets), text=f"Completed {target}")

    progress.empty()
    status_box.empty()
    return run_root, execution_rows


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
        runs.append(
            {
                "path": run_root,
                "targets": sorted(targets),
                "label": f"{rel_path}  |  {len(targets)} target(s)  |  {updated_at.strftime('%Y-%m-%d %H:%M')}",
                "updated_at": updated_at,
            }
        )

    return sorted(runs, key=lambda item: item["updated_at"], reverse=True)


@st.cache_data(show_spinner=False)
def load_summary(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict[str, Any]:
    return load_json(path)


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def primary_metric_name(task_type: str, summary_df: pd.DataFrame) -> str:
    candidates = {
        "binary": ["roc_auc", "average_precision", "recall", "accuracy"],
        "categorical": ["f1_macro", "accuracy"],
        "continuous": ["r2", "rmse", "mae"],
    }
    for metric in candidates.get(task_type, []):
        if metric in summary_df.columns:
            return metric
    numeric_candidates = [
        column
        for column in summary_df.columns
        if pd.api.types.is_numeric_dtype(summary_df[column]) and column not in {"fit_seconds"}
    ]
    return numeric_candidates[0] if numeric_candidates else ""


def format_metric_value(metric_name: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if metric_name in {"fit_seconds"}:
        return f"{float(value):.1f}s"
    return f"{float(value):.3f}"


def build_metric_bar_chart(summary_df: pd.DataFrame, task_type: str):
    metric = primary_metric_name(task_type, summary_df)
    if not metric or metric not in summary_df.columns:
        return None, ""

    if "status" in summary_df.columns:
        df = summary_df[summary_df["status"] == "ok"].copy()
    else:
        df = summary_df.copy()
    if df.empty:
        return None, metric

    ascending = metric in {"rmse", "mae"}
    df = df.sort_values(metric, ascending=ascending)

    sns.set_theme(style="dark")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#0d1727")

    palette = ["#5eead4", "#60a5fa", "#38bdf8", "#f59e0b", "#f472b6", "#c084fc"]
    sns.barplot(
        data=df,
        x=metric,
        y="model",
        hue="model",
        dodge=False,
        palette=palette[: len(df)],
        ax=ax,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.set_title(f"Model comparison by {metric}", color="white", fontsize=13, pad=12)
    ax.set_xlabel(metric, color="#dbeafe")
    ax.set_ylabel("")
    ax.tick_params(colors="#dbeafe")
    for spine in ax.spines.values():
        spine.set_color("#1f314f")
    ax.grid(axis="x", color="#24364f", alpha=0.4)
    fig.tight_layout()
    return fig, metric


def build_metric_heatmap(summary_df: pd.DataFrame, task_type: str):
    candidates = {
        "binary": [
            "roc_auc",
            "average_precision",
            "recall",
            "precision",
            "specificity",
            "f_beta",
            "accuracy",
        ],
        "categorical": ["f1_macro", "accuracy", "roc_auc_ovr"],
        "continuous": ["r2", "rmse", "mae"],
    }
    metrics = [metric for metric in candidates.get(task_type, []) if metric in summary_df.columns]
    if not metrics:
        return None

    if "status" in summary_df.columns:
        df = summary_df[summary_df["status"] == "ok"][["model", *metrics]].copy()
    else:
        df = summary_df[["model", *metrics]].copy()
    if df.empty:
        return None

    df = df.set_index("model")
    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.1), max(2.5, len(df) * 0.65)))
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#0d1727")
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap=sns.color_palette(["#0f172a", "#1d4ed8", "#2dd4bf"], as_cmap=True),
        linewidths=0.6,
        linecolor="#14233b",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Metric matrix", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#dbeafe", labelrotation=0)
    fig.tight_layout()
    return fig


def aggregate_importances(target_dir: Path) -> pd.DataFrame:
    frames = []
    for model_dir in sorted(path for path in target_dir.iterdir() if path.is_dir()):
        fi_path = model_dir / "feature_importance.csv"
        if not fi_path.exists():
            continue
        frame = load_feature_importance(str(fi_path))
        if frame.empty:
            continue
        frame["Model"] = model_dir.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    aggregated = (
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
    return aggregated


def build_importance_chart(aggregated: pd.DataFrame):
    if aggregated.empty:
        return None

    top = aggregated.head(15).sort_values("mean_abs_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4.2, len(top) * 0.35)))
    fig.patch.set_facecolor("#08111f")
    ax.set_facecolor("#0d1727")
    ax.barh(
        top["Feature"],
        top["mean_abs_importance"],
        color="#5eead4",
        alpha=0.85,
        edgecolor="#99f6e4",
    )
    ax.set_title("Cross-model permutation importance", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Mean absolute importance", color="#dbeafe")
    ax.tick_params(colors="#dbeafe")
    for spine in ax.spines.values():
        spine.set_color("#1f314f")
    ax.grid(axis="x", color="#24364f", alpha=0.35)
    fig.tight_layout()
    return fig


def render_run_results(run_root: Path) -> None:
    target_dirs = sorted(
        [
            child
            for child in run_root.iterdir()
            if child.is_dir()
            and child.name != "_runtime"
            and (child / "metadata.json").exists()
            and (child / "benchmark_summary.csv").exists()
        ],
        key=lambda path: path.name,
    )

    if not target_dirs:
        st.warning("No finished target outputs were found in this run.")
        return

    rel = run_root.relative_to(BASE_DIR)
    st.markdown(
        f'<div class="panel-note"><strong>Results root:</strong> <code>{rel}</code></div>',
        unsafe_allow_html=True,
    )

    target_tabs = st.tabs([target_dir.name for target_dir in target_dirs])
    for tab, target_dir in zip(target_tabs, target_dirs, strict=False):
        with tab:
            metadata = load_metrics(str(target_dir / "metadata.json"))
            summary_df = load_summary(str(target_dir / "benchmark_summary.csv"))
            task_type = metadata.get("task_type", "binary")
            if not summary_df.empty and "status" in summary_df.columns:
                ok_rows = summary_df[summary_df["status"] == "ok"].copy()
            else:
                ok_rows = summary_df.copy()
            selected_metric = primary_metric_name(task_type, summary_df) if not summary_df.empty else ""

            top_row = st.columns(4)
            with top_row[0]:
                metric_card(
                    "Target",
                    target_dir.name,
                    metadata.get("split_strategy", "unknown").upper(),
                )
            with top_row[1]:
                feature_count = len(metadata.get("data_configuration", {}).get("input_features", []))
                metric_card("Features", str(feature_count), "Model input columns")
            with top_row[2]:
                metric_card(
                    "Models",
                    str(len(ok_rows)),
                    "Successful trainings",
                )
            with top_row[3]:
                if not ok_rows.empty and selected_metric in ok_rows.columns:
                    best_idx = ok_rows[selected_metric].idxmin() if selected_metric in {"rmse", "mae"} else ok_rows[selected_metric].idxmax()
                    best_value = ok_rows.loc[best_idx, selected_metric]
                    metric_card(
                        f"Best {selected_metric}",
                        format_metric_value(selected_metric, best_value),
                        str(ok_rows.loc[best_idx, "model"]).upper(),
                    )
                else:
                    metric_card("Best metric", "n/a", "No successful run")

            st.markdown("")
            render_section_heading(
                "Model Comparison",
                "Use the summary table and charts to compare the current feature subset across selected models.",
            )

            chart_col, heatmap_col = st.columns([0.95, 1.05], gap="large")
            with chart_col:
                bar_fig, metric_name = build_metric_bar_chart(summary_df, task_type)
                if bar_fig is not None:
                    render_figure_with_downloads(bar_fig, f"{slugify(target_dir.name)}-model-comparison-{metric_name or 'metric'}")
                    plt.close(bar_fig)
                else:
                    st.info("No comparable metric plot available for this target.")
            with heatmap_col:
                heatmap_fig = build_metric_heatmap(summary_df, task_type)
                if heatmap_fig is not None:
                    render_figure_with_downloads(heatmap_fig, f"{slugify(target_dir.name)}-metric-matrix")
                    plt.close(heatmap_fig)
                else:
                    st.info("No metric matrix available for this target.")

            if not summary_df.empty:
                preferred_columns = [
                    column
                    for column in [
                        "model",
                        "status",
                        "roc_auc",
                        "average_precision",
                        "recall",
                        "precision",
                        "specificity",
                        "f_beta",
                        "accuracy",
                        "r2",
                        "rmse",
                        "mae",
                        "fit_seconds",
                    ]
                    if column in summary_df.columns
                ]
                st.dataframe(
                    summary_df[preferred_columns] if preferred_columns else summary_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("No benchmark summary found for this target.")

            render_section_heading(
                "Feature Pruning View",
                "Permutation importance is aggregated across models to highlight consistently weak variables.",
            )
            aggregated = aggregate_importances(target_dir)
            imp_left, imp_right = st.columns([1.05, 0.95], gap="large")
            with imp_left:
                importance_fig = build_importance_chart(aggregated)
                if importance_fig is not None:
                    render_figure_with_downloads(importance_fig, f"{slugify(target_dir.name)}-cross-model-permutation-importance")
                    plt.close(importance_fig)
                else:
                    st.info("Feature importance was not produced for this target.")
            with imp_right:
                if not aggregated.empty:
                    weakest = aggregated.sort_values("mean_abs_importance", ascending=True).head(10)
                    st.dataframe(
                        weakest.rename(
                            columns={
                                "Feature": "Candidate to remove",
                                "mean_importance": "Mean importance",
                                "mean_abs_importance": "Mean abs importance",
                                "models_reported": "Models",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No pruning candidates available yet.")

            render_section_heading(
                "Per-Model Details",
                "Inspect saved plots and detailed metrics for each trained model.",
            )
            combined_curves = [
                target_dir / "combined_roc_curve.png",
                target_dir / "combined_pr_curve.png",
            ]
            available_combined = [path for path in combined_curves if path.exists()]
            if available_combined:
                st.markdown(
                    '<div class="panel-note">Combined ROC and precision-recall curves across all trained models for this target.</div>',
                    unsafe_allow_html=True,
                )
                combined_cols = st.columns(2)
                for idx, plot_path in enumerate(available_combined):
                    with combined_cols[idx % 2]:
                        render_saved_plot(
                            plot_path,
                            plot_path.stem.replace("_", " ").title(),
                        )

            model_names = []
            if not summary_df.empty and "model" in summary_df.columns:
                model_names = summary_df["model"].tolist()
            else:
                model_names = sorted([child.name for child in target_dir.iterdir() if child.is_dir()])

            if model_names:
                model_tabs = st.tabs([model_name.upper() for model_name in model_names])
                for model_tab, model_name in zip(model_tabs, model_names, strict=False):
                    with model_tab:
                        model_dir = target_dir / model_name
                        metrics = load_metrics(str(model_dir / "metrics.json"))
                        metrics_cols = st.columns(4)
                        if metrics:
                            scalar_metrics = [
                                key
                                for key, value in metrics.items()
                                if isinstance(value, (int, float)) and key != "confusion_matrix"
                            ]
                            for column, metric_key in zip(metrics_cols, scalar_metrics[:4], strict=False):
                                with column:
                                    metric_card(metric_key, format_metric_value(metric_key, metrics[metric_key]), "Saved metric")
                        else:
                            st.info("No metrics.json found for this model.")

                        plot_files = [
                            model_dir / "roc_curve.png",
                            model_dir / "pr_curve.png",
                            model_dir / "confusion_matrix.png",
                            model_dir / "feature_importance.png",
                            model_dir / "actual_vs_predicted.png",
                        ]
                        available_plots = [path for path in plot_files if path.exists()]
                        if available_plots:
                            gallery = st.columns(2)
                            for idx, plot_path in enumerate(available_plots):
                                with gallery[idx % 2]:
                                    render_saved_plot(
                                        plot_path,
                                        plot_path.stem.replace("_", " ").title(),
                                    )

                        with st.expander("Raw metrics payload", expanded=False):
                            st.json(metrics)

                        fi_frame = load_feature_importance(str(model_dir / "feature_importance.csv"))
                        if not fi_frame.empty:
                            st.dataframe(fi_frame.head(20), use_container_width=True, hide_index=True)


def render_launchpad(base_config: dict[str, Any]) -> None:
    all_features = list(base_config.get("input_features", []))
    init_feature_state(all_features)

    targets = available_targets()
    if not targets:
        st.error("No grid-search outputs were found under `gridsearch/preoperative/`.")
        return

    if "selected_targets" not in st.session_state:
        st.session_state.selected_targets = list(targets)

    if "run_name" not in st.session_state:
        st.session_state.run_name = ""

    if "selected_features_picker" not in st.session_state:
        st.session_state.selected_features_picker = list(st.session_state.selected_features)

    render_section_heading(
        "Launchpad",
        "A simpler workflow: choose outcomes, confirm the models, shape the feature set, then launch the run.",
    )

    setup_cols = st.columns(3, gap="large")
    with setup_cols[0]:
        st.markdown('<div class="panel-note"><strong>Step 1</strong><br/>Choose the outcome targets.</div>', unsafe_allow_html=True)
        selected_targets = st.multiselect(
            "Outcome targets",
            targets,
            default=[target for target in st.session_state.selected_targets if target in targets] or targets,
            help="Each target is trained separately with its own saved best parameters.",
        )
        st.session_state.selected_targets = selected_targets
        target_actions = st.columns(2)
        if target_actions[0].button("All Targets", use_container_width=True):
            st.session_state.selected_targets = list(targets)
            st.rerun()
        if target_actions[1].button("Clear Targets", use_container_width=True):
            st.session_state.selected_targets = []
            st.rerun()

    available_models = model_union_for_targets(selected_targets or targets)
    default_models = st.session_state.get("selected_models", available_models)
    default_models = [model for model in default_models if model in available_models] or available_models

    with setup_cols[1]:
        st.markdown('<div class="panel-note"><strong>Step 2</strong><br/>Pick the learners you want to compare.</div>', unsafe_allow_html=True)
        selected_models = st.multiselect(
            "Models",
            available_models,
            default=default_models,
            help="If a chosen target does not have saved tuned parameters for a model, that pair is skipped automatically.",
        )
        st.session_state.selected_models = selected_models
        model_actions = st.columns(2)
        if model_actions[0].button("All Models", use_container_width=True):
            st.session_state.selected_models = list(available_models)
            st.rerun()
        if model_actions[1].button("Clear Models", use_container_width=True):
            st.session_state.selected_models = []
            st.rerun()

    with setup_cols[2]:
        st.markdown('<div class="panel-note"><strong>Step 3</strong><br/>Name the run and adjust the training policy if needed.</div>', unsafe_allow_html=True)
        st.text_input(
            "Run label",
            key="run_name",
            placeholder="e.g. symptom-lite-temporal",
            help="Used to name the result folder under `outputs/studio_runs/`.",
        )
        with st.expander("Advanced training settings", expanded=False):
            split_strategy = st.selectbox(
                "Split strategy",
                ["temporal", "random", "predefined"],
                index=0,
            )
            test_size = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.01)
            threshold_val_size = st.slider(
                "Threshold validation size",
                min_value=0.05,
                max_value=0.40,
                value=0.20,
                step=0.01,
            )
            min_recall = st.slider("Minimum recall", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
            f_beta = st.slider("F-beta", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            fn_cost = st.slider("False-negative cost", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
            fp_cost = st.slider("False-positive cost", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
            split_column = st.text_input("Predefined split column", value="Split")
            date_column = st.text_input("Temporal sort column", value="Date of surgery")

    if "split_strategy" not in locals():
        split_strategy = "temporal"
        test_size = 0.20
        threshold_val_size = 0.20
        min_recall = 0.90
        f_beta = 2.0
        fn_cost = 5.0
        fp_cost = 1.0
        split_column = "Split"
        date_column = "Date of surgery"

    reduction_pct = 0.0
    if all_features:
        reduction_pct = 100.0 * (1.0 - (len(st.session_state.selected_features) / len(all_features)))
    represented_groups = len({feature_group(feature) for feature in st.session_state.selected_features})

    snapshot_cols = st.columns(5)
    with snapshot_cols[0]:
        metric_card("Targets", str(len(selected_targets)), "Selected outcomes")
    with snapshot_cols[1]:
        metric_card("Models", str(len(selected_models)), "Requested learners")
    with snapshot_cols[2]:
        metric_card("Features", str(len(st.session_state.selected_features)), "Current subset")
    with snapshot_cols[3]:
        metric_card("Reduction", f"{reduction_pct:.0f}%", "Vs. full preoperative pool")
    with snapshot_cols[4]:
        metric_card("Groups", str(represented_groups), "Clinical areas covered")

    with st.expander("Target-by-target model availability", expanded=False):
        availability_df = build_target_model_availability(selected_targets)
        if not availability_df.empty:
            st.dataframe(availability_df, use_container_width=True, hide_index=True)

    st.markdown("")
    render_section_heading(
        "Feature Builder",
        "Use presets for fast starting points, then refine the exact subset with a single searchable selector and optional group actions.",
    )

    preset_cols = st.columns(5)
    if preset_cols[0].button("Full Preoperative", use_container_width=True):
        apply_feature_preset("all", all_features)
        st.rerun()
    if preset_cols[1].button("Clinical Core", use_container_width=True):
        apply_feature_preset("clinical", all_features)
        st.rerun()
    if preset_cols[2].button("Imaging Focus", use_container_width=True):
        apply_feature_preset("imaging", all_features)
        st.rerun()
    if preset_cols[3].button("Compact Seed", use_container_width=True):
        apply_feature_preset("compact", all_features)
        st.rerun()
    if preset_cols[4].button("Clear", use_container_width=True):
        apply_feature_preset("clear", all_features)
        st.rerun()

    builder_cols = st.columns([1.18, 0.82], gap="large")
    group_names = sorted({feature_group(feature) for feature in all_features})
    group_map = {group: [feature for feature in all_features if feature_group(feature) == group] for group in group_names}

    with builder_cols[0]:
        focus_group = st.selectbox(
            "Quick group actions",
            options=group_names,
            help="Use this to add, remove, or isolate an entire clinical feature group.",
        )
        group_actions = st.columns(3)
        focus_features = group_map.get(focus_group, [])
        if group_actions[0].button("Add Group", use_container_width=True):
            set_selected_features(st.session_state.selected_features + focus_features, all_features)
            st.rerun()
        if group_actions[1].button("Remove Group", use_container_width=True):
            set_selected_features(
                [feature for feature in st.session_state.selected_features if feature not in set(focus_features)],
                all_features,
            )
            st.rerun()
        if group_actions[2].button("Only This Group", use_container_width=True):
            set_selected_features(focus_features, all_features)
            st.rerun()

        selected_features = st.multiselect(
            "Selected input features",
            options=all_features,
            default=st.session_state.selected_features,
            key="selected_features_picker",
            help="This is the main feature selection control. Use typing to search quickly.",
        )
        st.session_state.selected_features = [f for f in all_features if f in set(selected_features)]

        group_tabs = st.tabs(group_names)
        for group_tab, group_name in zip(group_tabs, group_names, strict=False):
            with group_tab:
                features_in_group = group_map[group_name]
                st.markdown(
                    f'<div class="panel-note"><strong>{group_name}</strong><br/>{len(features_in_group)} available features.</div>',
                    unsafe_allow_html=True,
                )
                render_feature_chips(features_in_group, limit=999)

    with builder_cols[1]:
        st.markdown(
            '<div class="panel-note"><strong>Selected subset</strong><br/>Use this preview to judge burden, balance across groups, and what might be removable next.</div>',
            unsafe_allow_html=True,
        )
        render_feature_chips(st.session_state.selected_features)

        group_counts = (
            pd.Series([feature_group(feature) for feature in st.session_state.selected_features])
            .value_counts()
            .rename_axis("Group")
            .reset_index(name="Selected")
        )
        if not group_counts.empty:
            st.dataframe(group_counts, use_container_width=True, hide_index=True)

        distribution = (
            pd.Series([feature_group(feature) for feature in st.session_state.selected_features])
            .value_counts()
            .sort_values(ascending=True)
        )
        if not distribution.empty:
            fig, ax = plt.subplots(figsize=(6.4, max(3.0, len(distribution) * 0.55)))
            fig.patch.set_facecolor("#08111f")
            ax.set_facecolor("#0d1727")
            ax.barh(distribution.index, distribution.values, color="#60a5fa", alpha=0.9)
            ax.set_title("Selected features by group", color="white", fontsize=13, pad=12)
            ax.tick_params(colors="#dbeafe")
            ax.set_xlabel("Count", color="#dbeafe")
            for spine in ax.spines.values():
                spine.set_color("#1f314f")
            ax.grid(axis="x", color="#24364f", alpha=0.35)
            fig.tight_layout()
            render_figure_with_downloads(fig, "selected-features-by-group")
            plt.close(fig)

    if not selected_targets:
        st.warning("Select at least one target before launching training.")
    if not selected_models:
        st.warning("Select at least one model before launching training.")
    if not st.session_state.selected_features:
        st.warning("Select at least one input feature before launching training.")

    st.markdown("")
    render_section_heading(
        "Launch Training",
        "Run the selected targets and models on the chosen subset. Results are written under `outputs/studio_runs/` and opened immediately below.",
    )
    launch_disabled = not (selected_targets and selected_models and st.session_state.selected_features)
    if st.button("Train Selected Configuration", type="primary", use_container_width=True, disabled=launch_disabled):
        with st.spinner("Training models on the selected feature subset..."):
            run_root, execution_rows = launch_training_run(
                targets=selected_targets,
                models=selected_models,
                selected_features=st.session_state.selected_features,
                base_config=base_config,
                run_name=st.session_state.run_name,
                split_strategy=split_strategy,
                test_size=test_size,
                split_column=split_column,
                date_column=date_column,
                threshold_val_size=threshold_val_size,
                min_recall=min_recall,
                f_beta=f_beta,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )

        st.cache_data.clear()
        st.session_state.latest_run_root = str(run_root)
        st.session_state.selected_run_root = str(run_root)

        execution_df = pd.DataFrame(execution_rows)
        ok_count = int((execution_df["status"] == "ok").sum()) if not execution_df.empty else 0
        fail_count = int((execution_df["status"] == "failed").sum()) if not execution_df.empty else 0
        if fail_count == 0:
            st.success(f"Training finished. Successful target runs: {ok_count}.")
        else:
            st.warning(f"Training finished with {fail_count} failed target run(s).")

        st.dataframe(execution_df, use_container_width=True, hide_index=True)

        with st.expander("Console logs", expanded=fail_count > 0):
            for row in execution_rows:
                log_file = row.get("log_file")
                if not log_file:
                    continue
                st.markdown(f"**{row['target']}**")
                try:
                    content = Path(log_file).read_text(encoding="utf-8")
                except OSError:
                    content = "Unable to read log file."
                st.code(content or "(empty log)", language="text")

        render_section_heading(
            "Latest Run",
            "The freshly trained outputs are rendered below so you can immediately inspect model behavior and importance profiles.",
        )
        render_run_results(run_root)


def render_results_explorer() -> None:
    runs = discover_runs()
    if not runs:
        st.info("No experiment outputs with metadata were found yet.")
        return

    default_root = st.session_state.get("selected_run_root")
    default_index = 0
    if default_root:
        for idx, run in enumerate(runs):
            if str(run["path"]) == default_root:
                default_index = idx
                break

    selected_label = st.selectbox(
        "Browse experiment outputs",
        options=[run["label"] for run in runs],
        index=default_index,
    )
    selected_run = next(run for run in runs if run["label"] == selected_label)
    st.session_state.selected_run_root = str(selected_run["path"])
    render_run_results(selected_run["path"])


def main() -> None:
    inject_theme()
    base_config = load_base_feature_config()

    all_targets = available_targets()
    all_features = base_config.get("input_features", [])
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Training Studio</div>
            <h1>MedModel Interface for Feature Reduction</h1>
            <p>
                Build reduced clinical feature sets, launch training with the project’s saved
                grid-search optima, and review outcome metrics plus permutation importance in one place.
                The interface is tuned for iterative pruning: choose a leaner subset, run the models,
                inspect what matters, and tighten the feature burden again.
            </p>
            <div class="micro-grid">
                <div class="micro-card">
                    <div class="label">Available targets</div>
                    <div class="value">{len(all_targets)}</div>
                </div>
                <div class="micro-card">
                    <div class="label">Max feature pool</div>
                    <div class="value">{len(all_features)}</div>
                </div>
                <div class="micro-card">
                    <div class="label">Saved tuning source</div>
                    <div class="value">Grid Search</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Launchpad", "Results Explorer"])
    with tabs[0]:
        render_launchpad(base_config)
    with tabs[1]:
        render_results_explorer()


if __name__ == "__main__":
    main()
