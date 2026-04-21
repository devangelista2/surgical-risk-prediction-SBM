from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
OUTPUTS_ROOT = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configs"
DOCTOR_SETTINGS_PATH = CONFIGS_DIR / "doctor_settings.json"

# Must be on sys.path so joblib can deserialise custom sklearn wrappers
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402 – after path setup

app = Flask(__name__)

# ── In-memory pipeline cache ───────────────────────────────────────────────────
_pipeline_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()

# ── Default doctor settings ────────────────────────────────────────────────────
DEFAULT_SETTINGS: dict[str, Any] = {
    "low_threshold": 0.30,
    "high_threshold": 0.60,
    "default_run_path": "",
    "hidden_runs": [],
}


def load_doctor_settings() -> dict[str, Any]:
    if DOCTOR_SETTINGS_PATH.exists():
        try:
            with open(DOCTOR_SETTINGS_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_doctor_settings(settings: dict[str, Any]) -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCTOR_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# ── Known field metadata (unit, range, default, hints) ────────────────────────
FIELD_META: dict[str, dict] = {
    "Age": {
        "unit": "years", "min": 0, "max": 120, "step": 1, "default": 60,
    },
    "Sex": {
        "unit": "", "type_override": "sex",
        "options": ["Female", "Male"], "option_values": [0, 1], "default": 0,
    },
    "Pre-Op KPS": {
        "unit": "%", "min": 0, "max": 100, "step": 10, "default": 80,
        "hint": "Karnofsky Performance Status — 100 = normal activity, 0 = deceased",
    },
    "ASA": {
        "unit": "", "min": 1, "max": 5, "step": 1, "default": 2,
        "hint": "1 = healthy  ·  2 = mild disease  ·  3 = severe  ·  4 = life-threatening  ·  5 = moribund",
    },
    "Charlson Comorbidity Index": {
        "unit": "", "min": 0, "max": 20, "step": 1, "default": 2,
    },
    "Radio_Pre-Op max_axial_diam_mm": {
        "unit": "mm", "min": 0, "max": 200, "step": 0.5, "default": 40.0,
    },
    "Radio_Midline shift entity": {
        "unit": "mm", "min": 0, "max": 30, "step": 0.5, "default": 0.0,
    },
    "Simpson Grade": {
        "unit": "", "min": 1, "max": 5, "step": 1, "default": 2,
        "hint": "1 = complete  ·  2 = coagulated dura  ·  3 = no coagulation  ·  4 = subtotal  ·  5 = decompression",
    },
    "WHO grade": {
        "unit": "", "min": 1, "max": 4, "step": 1, "default": 1,
    },
    "ICA compression": {
        "unit": "", "type_override": "boolean", "default": 0,
    },
    "ICA encasement": {
        "unit": "", "type_override": "boolean", "default": 0,
    },
    "Complete asportation": {
        "unit": "", "type_override": "boolean", "default": 0,
    },
}

# Manual group assignment (overrides the heuristics below)
FIELD_GROUPS: dict[str, str] = {
    "Age": "Demographics",
    "Sex": "Demographics",
    "Date of surgery": "Demographics",
    "Date of Birth": "Demographics",
    "Pre-Op KPS": "Functional Status",
    "ASA": "Functional Status",
    "Charlson Comorbidity Index": "Functional Status",
    "Center": "Administrative",
    "Simpson Grade": "Surgical Details",
    "EOR": "Surgical Details",
    "Surgical Approach": "Surgical Details",
    "WHO grade": "Surgical Details",
    "Complete asportation": "Surgical Details",
    "ICA compression": "Vascular Involvement",
    "ICA encasement": "Vascular Involvement",
    "Cavernous sinus involvement": "Vascular Involvement",
    "ACA/MCA/PCA involvement": "Vascular Involvement",
    "MCA compression": "Vascular Involvement",
    "MCA encasement": "Vascular Involvement",
    "ACA compression": "Vascular Involvement",
    "ACA encasement": "Vascular Involvement",
    "SCA, AICA, PICA involvement": "Vascular Involvement",
    "Brainstem compression": "Cranial Nerves & Brainstem",
    "CNs III/IV/VI involvement": "Cranial Nerves & Brainstem",
    "CNs V/VII/VIII involvement": "Cranial Nerves & Brainstem",
    "CNs IX, X, XI involvement": "Cranial Nerves & Brainstem",
}

# Boolean detection helpers
BOOLEAN_PREFIXES = ("Comorbidities_", "Symptoms_")
BOOLEAN_SUBSTRINGS = (
    "Radio_Ventricle",
    "Radio_Calcification",
    "Radio_Edema",
    "Radio_Dural_tail",
    "Radio_Hyperostosis",
    "Radio_Bone_invasion",
    "Radio_Cystic_necrosis",
    "Optic Canal",
    "Optic nerve",
    "ICA/Syphon",
)

# Human-readable target labels and descriptions
TARGET_LABELS: dict[str, str] = {
    "complications_30d": "30-Day Complications",
    "KPS_Discharge Worsened": "KPS Worsening at Discharge",
    "New neurological deficits": "New Neurological Deficits",
    "Severe complication": "Severe Complication",
}
TARGET_DESCRIPTIONS: dict[str, str] = {
    "complications_30d": "Probability of any complication occurring within 30 days of surgery",
    "KPS_Discharge Worsened": "Probability of functional status deterioration at hospital discharge",
    "New neurological deficits": "Probability of new post-operative neurological deficits",
    "Severe complication": "Probability of a severe or life-threatening complication",
}

# Model loading priority (safest / most portable first)
MODEL_PRIORITY = ["hgb", "rf", "lr", "svc", "torch_mlp", "torch_ft_transformer", "ridge"]

MODEL_LABELS: dict[str, str] = {
    "hgb": "Hist. Gradient Boosting",
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "svc": "SVM (RBF)",
    "torch_mlp": "Neural Network (MLP)",
    "torch_ft_transformer": "FT-Transformer",
    "ridge": "Ridge Regression",
}

# Desired display order for feature groups
GROUP_ORDER = [
    "Demographics",
    "Functional Status",
    "Comorbidities",
    "Presenting Symptoms",
    "Imaging Findings",
    "Vascular Involvement",
    "Cranial Nerves & Brainstem",
    "Surgical Details",
    "Administrative",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_label(feature: str) -> str:
    """Strip known prefixes and clean underscores for display."""
    for prefix in ("Comorbidities_", "Symptoms_", "Radio_"):
        if feature.startswith(prefix):
            feature = feature[len(prefix):]
            break
    return feature.replace("_", " ").strip()


def get_field_type(feature: str, cols_string: list, cols_date: list, cols_multi: list) -> str:
    """Determine the UI input type for a feature column."""
    meta = FIELD_META.get(feature, {})
    if "type_override" in meta:
        return meta["type_override"]
    if feature in cols_date:
        return "date"
    if feature in cols_string:
        return "select"
    if feature in cols_multi:
        return "multi"
    if feature.startswith(BOOLEAN_PREFIXES):
        return "boolean"
    for kw in BOOLEAN_SUBSTRINGS:
        if kw in feature:
            return "boolean"
    return "numeric"


def get_field_group(feature: str) -> str:
    """Return the display section for a feature."""
    if feature in FIELD_GROUPS:
        return FIELD_GROUPS[feature]
    if feature.startswith("Comorbidities_"):
        return "Comorbidities"
    if feature.startswith("Symptoms_"):
        return "Presenting Symptoms"
    if feature.startswith("Radio_") or feature.startswith("Optic") or "ICA" in feature:
        return "Imaging Findings"
    if "KPS" in feature:
        return "Functional Status"
    return "Other"


def _load_pipeline_cached(pipeline_path: str) -> Any:
    """Load a pipeline from disk with in-memory caching."""
    with _cache_lock:
        if pipeline_path in _pipeline_cache:
            return _pipeline_cache[pipeline_path]

    pipeline = joblib.load(pipeline_path)

    with _cache_lock:
        _pipeline_cache[pipeline_path] = pipeline

    return pipeline


def _extract_string_options(pipeline: Any, cols_string: list[str]) -> dict[str, list[str]]:
    """Extract fitted OneHotEncoder categories from the pipeline."""
    options: dict[str, list[str]] = {}
    try:
        ct = pipeline.named_steps.get("preprocess")
        if ct is None:
            return options
        for name, transformer, columns in ct.transformers_:
            if name == "categorical":
                ohe = transformer.named_steps.get("onehot")
                if ohe is None:
                    continue
                for col, cats in zip(columns, ohe.categories_):
                    options[col] = [str(c) for c in cats.tolist()]
    except Exception:
        pass
    return options


def _best_model_name(target_dir: Path) -> str | None:
    """Pick the best model by ROC-AUC from benchmark_summary.csv."""
    summary = target_dir / "benchmark_summary.csv"
    if not summary.exists():
        return None
    try:
        df = pd.read_csv(summary)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        if df.empty:
            return None
        if "roc_auc" in df.columns:
            return str(df.loc[df["roc_auc"].idxmax(), "model"])
        if "model" in df.columns:
            return str(df.iloc[0]["model"])
    except Exception:
        pass
    return None


def get_target_models(run_path: str, target_name: str) -> list[dict]:
    """Return all available models for a target with their ROC-AUC scores."""
    target_dir = Path(run_path) / target_name
    if not target_dir.exists():
        return []

    roc_map: dict[str, float | None] = {}
    best_model = _best_model_name(target_dir)
    summary = target_dir / "benchmark_summary.csv"
    if summary.exists():
        try:
            df = pd.read_csv(summary)
            if "status" in df.columns:
                df = df[df["status"] == "ok"]
            if "roc_auc" in df.columns and "model" in df.columns:
                for _, row in df.iterrows():
                    roc_map[str(row["model"])] = float(row["roc_auc"])
        except Exception:
            pass

    models = []
    for model_dir in sorted(target_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "pipeline.joblib").exists():
            continue
        mname = model_dir.name
        models.append({
            "model": mname,
            "label": MODEL_LABELS.get(mname, mname.upper()),
            "roc_auc": roc_map.get(mname),
            "is_best": mname == best_model,
        })

    models.sort(key=lambda x: x["roc_auc"] or 0, reverse=True)
    return models


def _find_loadable_pipeline(target_dir: Path) -> tuple[str | None, Any]:
    """
    Try to load the best pipeline for a target, falling back through the
    priority list if the best model can't be deserialised (e.g. missing torch).
    """
    best = _best_model_name(target_dir)
    candidates = []
    if best:
        candidates.append(best)
    for m in MODEL_PRIORITY:
        if m not in candidates:
            candidates.append(m)

    for model_name in candidates:
        p = target_dir / model_name / "pipeline.joblib"
        if not p.exists():
            continue
        try:
            return model_name, _load_pipeline_cached(str(p))
        except Exception:
            continue

    return None, None


# ── Preset discovery ───────────────────────────────────────────────────────────

def _is_valid_run(run_root: Path) -> bool:
    if run_root.name.startswith("_"):
        return False
    return any(
        (td / "metadata.json").exists() and list(td.glob("*/pipeline.joblib"))
        for td in run_root.iterdir()
        if td.is_dir() and td.name != "_runtime"
    )


def discover_presets(include_hidden: bool = False) -> list[dict[str, Any]]:
    """
    Return available preset configurations, each tagged with a ``hidden`` flag.
    Pass ``include_hidden=True`` to include hidden presets (used by the settings panel).
    """
    hidden_set = set(load_doctor_settings().get("hidden_runs", []))

    presets: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(run_root: Path, nice_name: str | None = None) -> None:
        key = str(run_root)
        if key in seen or not run_root.exists() or not _is_valid_run(run_root):
            return
        seen.add(key)

        target_dirs = [
            td for td in run_root.iterdir()
            if td.is_dir() and td.name != "_runtime"
            and (td / "metadata.json").exists()
            and list(td.glob("*/pipeline.joblib"))
        ]

        label = nice_name or (
            run_root.name.replace("_", " ").replace("-", " ").title()
        )

        presets.append({
            "id": key,
            "name": label,
            "run_path": key,
            "targets": sorted(td.name for td in target_dirs),
            "target_count": len(target_dirs),
            "hidden": key in hidden_set,
        })

    _add(OUTPUTS_ROOT / "preoperative_from_gridsearch", "Preoperative Standard Assessment")
    _add(OUTPUTS_ROOT / "gridsearch_eval", "Grid-Search Evaluation Set")

    studio_root = OUTPUTS_ROOT / "studio_runs"
    if studio_root.exists():
        for run in sorted(studio_root.iterdir(), key=lambda p: p.name, reverse=True)[:12]:
            if run.is_dir():
                _add(run)

    if include_hidden:
        return presets
    return [p for p in presets if not p["hidden"]]


# ── Preset metadata ────────────────────────────────────────────────────────────

def get_preset_metadata(run_path: str) -> dict[str, Any] | None:
    """
    Return the feature definitions (type, default, options …) and target list
    for a given preset run folder.
    """
    run_root = Path(run_path)
    if not run_root.exists():
        return None

    target_dirs = sorted([
        td for td in run_root.iterdir()
        if td.is_dir() and td.name != "_runtime"
        and (td / "metadata.json").exists()
        and list(td.glob("*/pipeline.joblib"))
    ])
    if not target_dirs:
        return None

    # Use the first target's metadata for the feature config
    with open(target_dirs[0] / "metadata.json", encoding="utf-8") as f:
        first_meta = json.load(f)

    data_cfg: dict[str, Any] = first_meta.get("data_configuration", {})
    input_features: list[str] = data_cfg.get("input_features", [])
    cols_string: list[str] = data_cfg.get("cols_string", [])
    cols_date: list[str] = data_cfg.get("cols_date", [])
    cols_multi: list[str] = data_cfg.get("cols_multi", [])

    # Try to get string-column options from the fitted encoder
    string_options: dict[str, list[str]] = {}
    model_name, pipeline = _find_loadable_pipeline(target_dirs[0])
    if pipeline is not None:
        string_options = _extract_string_options(pipeline, cols_string)

    # Build feature descriptors
    features: list[dict[str, Any]] = []
    for feat in input_features:
        ftype = get_field_type(feat, cols_string, cols_date, cols_multi)
        meta = FIELD_META.get(feat, {})
        group = get_field_group(feat)

        entry: dict[str, Any] = {
            "name": feat,
            "label": clean_label(feat),
            "type": ftype,
            "group": group,
            "unit": meta.get("unit", ""),
            "hint": meta.get("hint", ""),
            "default": meta.get("default"),
        }

        if ftype in ("numeric",):
            entry["min"] = meta.get("min", 0)
            entry["max"] = meta.get("max", 999)
            entry["step"] = meta.get("step", 1)
            if entry["default"] is None:
                entry["default"] = meta.get("min", 0)

        elif ftype == "sex":
            entry["options"] = meta.get("options", ["Female", "Male"])
            entry["option_values"] = meta.get("option_values", [0, 1])
            if entry["default"] is None:
                entry["default"] = 0

        elif ftype == "select":
            opts = string_options.get(feat, [])
            entry["options"] = opts
            if entry["default"] is None:
                entry["default"] = opts[0] if opts else ""

        elif ftype in ("boolean",):
            if entry["default"] is None:
                entry["default"] = 0

        elif ftype == "date":
            if entry["default"] is None:
                entry["default"] = ""

        features.append(entry)

    # Sort features by group order, then name
    def _sort_key(f: dict) -> tuple:
        g = f["group"]
        gi = GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
        return (gi, f["name"])

    features.sort(key=_sort_key)

    # Build target descriptors
    targets: list[dict[str, Any]] = []
    for td in target_dirs:
        tname = td.name
        bm, _ = _find_loadable_pipeline(td)
        threshold = 0.5
        if bm:
            policy_p = td / bm / "decision_policy.json"
            if policy_p.exists():
                try:
                    with open(policy_p, encoding="utf-8") as f:
                        policy = json.load(f)
                    threshold = float(policy.get("threshold", 0.5))
                except Exception:
                    pass

        targets.append({
            "name": tname,
            "label": TARGET_LABELS.get(tname, tname.replace("_", " ").title()),
            "description": TARGET_DESCRIPTIONS.get(tname, ""),
            "best_model": bm,
            "model_threshold": threshold,
        })

    return {
        "run_path": run_path,
        "features": features,
        "targets": targets,
    }


# ── Prediction ─────────────────────────────────────────────────────────────────

def run_prediction(
    run_path: str,
    feature_values: dict[str, Any],
    model_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run inference for every target in the preset and return risk probabilities."""
    run_root = Path(run_path)
    target_dirs = sorted([
        td for td in run_root.iterdir()
        if td.is_dir() and td.name != "_runtime"
        and (td / "metadata.json").exists()
        and list(td.glob("*/pipeline.joblib"))
    ])

    results: list[dict[str, Any]] = []

    for td in target_dirs:
        tname = td.name

        # Use caller-specified model if provided, otherwise pick the best one
        override = (model_overrides or {}).get(tname)
        if override:
            override_path = td / override / "pipeline.joblib"
            if override_path.exists():
                model_name = override
                try:
                    pipeline = _load_pipeline_cached(str(override_path))
                except Exception:
                    model_name, pipeline = _find_loadable_pipeline(td)
            else:
                model_name, pipeline = _find_loadable_pipeline(td)
        else:
            model_name, pipeline = _find_loadable_pipeline(td)

        if pipeline is None:
            results.append({
                "target": tname,
                "label": TARGET_LABELS.get(tname, tname),
                "error": "No loadable model found for this target.",
            })
            continue

        try:
            with open(td / "metadata.json", encoding="utf-8") as f:
                meta = json.load(f)
            data_cfg = meta.get("data_configuration", {})
            input_features: list[str] = data_cfg.get("input_features", [])
            cols_string: list[str] = data_cfg.get("cols_string", [])
            cols_date: list[str] = data_cfg.get("cols_date", [])

            # Build a single-row DataFrame
            row: dict[str, Any] = {}
            for feat in input_features:
                val = feature_values.get(feat)

                if val is None or val == "":
                    val = np.nan
                elif feat in cols_date:
                    val = str(val) if val else np.nan
                elif feat in cols_string:
                    val = str(val) if val else np.nan
                else:
                    # Numeric / boolean
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan

                row[feat] = val

            df_input = pd.DataFrame([row])[input_features]

            # Predict probability
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(df_input)
                model_obj = pipeline.named_steps.get("model")
                classes = getattr(model_obj, "classes_", [False, True])
                pos_idx = 1
                for i, c in enumerate(classes):
                    if c is True or c == 1 or str(c).lower() == "true":
                        pos_idx = i
                        break
                risk_prob = float(proba[0, pos_idx])
            else:
                pred = pipeline.predict(df_input)
                risk_prob = 1.0 if pred[0] else 0.0

            # Load model threshold
            threshold = 0.5
            policy_p = td / model_name / "decision_policy.json"
            if policy_p.exists():
                try:
                    with open(policy_p, encoding="utf-8") as f:
                        policy = json.load(f)
                    threshold = float(policy.get("threshold", 0.5))
                except Exception:
                    pass

            results.append({
                "target": tname,
                "label": TARGET_LABELS.get(tname, tname.replace("_", " ").title()),
                "description": TARGET_DESCRIPTIONS.get(tname, ""),
                "probability": risk_prob,
                "probability_pct": round(risk_prob * 100, 1),
                "model": model_name,
                "model_threshold": threshold,
            })

        except Exception as exc:
            results.append({
                "target": tname,
                "label": TARGET_LABELS.get(tname, tname),
                "error": str(exc),
            })

    return results


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def root_redirect():
    return redirect(url_for("doctor_index"))


@app.route("/doctor/")
@app.route("/doctor")
def doctor_index():
    return render_template("doctor.html")


@app.route("/api/doctor/presets")
def api_presets():
    # include_hidden=true is used by the settings panel to show everything
    include_hidden = request.args.get("include_hidden", "false").lower() == "true"
    presets = discover_presets(include_hidden=include_hidden)
    settings = load_doctor_settings()
    return jsonify({"presets": presets, "settings": settings})


@app.route("/api/doctor/presets/toggle-hidden", methods=["POST"])
def api_toggle_hidden():
    data = request.get_json(force=True) or {}
    run_path = data.get("run_path", "").strip()
    if not run_path:
        return jsonify({"error": "run_path is required"}), 400
    settings = load_doctor_settings()
    hidden = set(settings.get("hidden_runs", []))
    if run_path in hidden:
        hidden.discard(run_path)
    else:
        hidden.add(run_path)
    settings["hidden_runs"] = sorted(hidden)
    save_doctor_settings(settings)
    return jsonify({"hidden_runs": settings["hidden_runs"]})


@app.route("/api/doctor/preset-meta")
def api_preset_meta():
    run_path = request.args.get("run_path", "").strip()
    if not run_path:
        return jsonify({"error": "run_path is required"}), 400
    meta = get_preset_metadata(run_path)
    if meta is None:
        return jsonify({"error": "Preset not found or has no valid trained models"}), 404
    return jsonify(meta)


@app.route("/api/doctor/target-models")
def api_target_models():
    run_path = request.args.get("run_path", "").strip()
    target = request.args.get("target", "").strip()
    if not run_path or not target:
        return jsonify({"error": "run_path and target are required"}), 400
    return jsonify({"models": get_target_models(run_path, target)})


@app.route("/api/doctor/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    run_path = (data or {}).get("run_path", "")
    features = (data or {}).get("features", {})
    model_overrides = (data or {}).get("model_overrides", {})
    if not run_path:
        return jsonify({"error": "run_path is required"}), 400
    try:
        results = run_prediction(run_path, features, model_overrides)
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/doctor/settings", methods=["GET"])
def api_get_settings():
    return jsonify(load_doctor_settings())


@app.route("/api/doctor/settings", methods=["POST"])
def api_save_settings():
    data = request.get_json(force=True) or {}
    settings = load_doctor_settings()
    for key in ("low_threshold", "high_threshold", "default_run_path"):
        if key in data:
            settings[key] = data[key]
    save_doctor_settings(settings)
    return jsonify(settings)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)
