from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
OUTPUTS_ROOT = BASE_DIR / "outputs"
CATEGORY_OPTIONS_FILE = OUTPUTS_ROOT / "category_options.json"
APP_FEATURE_CONFIG_FILE = BASE_DIR / "configs" / "preoperative_reduced.json"

# Needed so joblib can resolve custom classes (e.g., torch models in src/nn).
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


st.set_page_config(page_title="MedModel Predictor", layout="wide")
st.title("MedModel: Risk Analysis")
st.caption("Decision-support tool (research). Not a replacement for clinical judgment.")


def ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


@st.cache_data
def load_json_file(path: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_metadata(path: str) -> dict[str, Any]:
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_category_options(path: str) -> dict[str, list[str]]:
    options_path = Path(path)
    if not options_path.exists():
        return {}

    with open(options_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    raw_columns: Any = payload
    if isinstance(payload, dict) and isinstance(payload.get("columns"), dict):
        raw_columns = payload["columns"]

    if not isinstance(raw_columns, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, values in raw_columns.items():
        if not isinstance(key, str):
            continue
        if not isinstance(values, list):
            continue
        cleaned = []
        for value in values:
            text = str(value).strip()
            if text:
                cleaned.append(text)
        normalized[key] = sorted(set(cleaned))
    return normalized


@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


def find_models(outputs_root: Path) -> dict[str, dict[str, Path]]:
    models: dict[str, dict[str, Path]] = {}
    if not outputs_root.exists():
        return models

    for pipeline_path in outputs_root.rglob("pipeline.joblib"):
        model_name = pipeline_path.parent.name
        target_name = pipeline_path.parent.parent.name
        models.setdefault(target_name, {})[model_name] = pipeline_path
    return models


def is_symptom_feature(name: str) -> bool:
    low_name = name.lower()
    return (
        low_name.startswith("symptoms_")
        or "compression" in low_name
        or "deficits" in low_name
        or "edema" in low_name
    )


def infer_features_from_pipeline(model: Any) -> list[str]:
    named_steps = getattr(model, "named_steps", {})
    preprocess = named_steps.get("preprocess")
    transformers = getattr(preprocess, "transformers", None) or getattr(
        preprocess, "transformers_", None
    )
    if not transformers:
        return []

    features: list[str] = []
    seen: set[str] = set()
    for transformer in transformers:
        if len(transformer) < 3:
            continue
        cols = transformer[2]
        if isinstance(cols, str):
            if cols not in {"drop", "passthrough"} and cols not in seen:
                features.append(cols)
                seen.add(cols)
            continue
        if isinstance(cols, (list, tuple)):
            for col in cols:
                if isinstance(col, str) and col not in seen:
                    features.append(col)
                    seen.add(col)
    return features


def derive_feature_config(
    data_config: dict[str, Any], model: Any
) -> tuple[list[str], list[str], list[str], list[str]]:
    input_features = ensure_list(data_config.get("input_features"))
    cols_string = ensure_list(data_config.get("cols_string"))
    cols_date = ensure_list(data_config.get("cols_date"))
    cols_multi = ensure_list(data_config.get("cols_multi"))

    ordered_features: list[str] = []
    seen: set[str] = set()
    for col in input_features + cols_string + cols_date + cols_multi:
        if col and col not in seen:
            ordered_features.append(col)
            seen.add(col)

    if not ordered_features:
        ordered_features = infer_features_from_pipeline(model)

    return ordered_features, cols_string, cols_date, cols_multi


def normalize_columns(raw_cols: Any) -> list[str]:
    if isinstance(raw_cols, str):
        if raw_cols in {"drop", "passthrough"}:
            return []
        return [raw_cols]
    if isinstance(raw_cols, (list, tuple)):
        return [col for col in raw_cols if isinstance(col, str)]
    return []


def first_step_with_attr(transformer: Any, attr_name: str) -> Any:
    if hasattr(transformer, attr_name):
        return transformer
    named_steps = getattr(transformer, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in named_steps.values():
            if hasattr(step, attr_name):
                return step
    return None


def infer_defaults_from_pipeline(model: Any) -> dict[str, Any]:
    named_steps = getattr(model, "named_steps", {})
    preprocess = named_steps.get("preprocess")
    transformers = getattr(preprocess, "transformers", None) or getattr(
        preprocess, "transformers_", None
    )
    if not transformers:
        return {}

    defaults: dict[str, Any] = {}
    for transformer_name, transformer_obj, raw_cols in transformers:
        if transformer_name == "remainder":
            continue
        cols = normalize_columns(raw_cols)
        if not cols:
            continue
        imputer = first_step_with_attr(transformer_obj, "statistics_")
        stats = list(getattr(imputer, "statistics_", [])) if imputer is not None else []
        if len(stats) != len(cols):
            continue
        for col, stat in zip(cols, stats):
            if pd.isna(stat):
                continue
            defaults[col] = stat
    return defaults


def infer_categories_from_pipeline(model: Any) -> dict[str, list[str]]:
    named_steps = getattr(model, "named_steps", {})
    preprocess = named_steps.get("preprocess")
    transformers = getattr(preprocess, "transformers", None) or getattr(
        preprocess, "transformers_", None
    )
    if not transformers:
        return {}

    categories_map: dict[str, list[str]] = {}
    for transformer_name, transformer_obj, raw_cols in transformers:
        if transformer_name == "remainder":
            continue
        cols = normalize_columns(raw_cols)
        if not cols:
            continue
        encoder = first_step_with_attr(transformer_obj, "categories_")
        categories = list(getattr(encoder, "categories_", [])) if encoder is not None else []
        if len(categories) != len(cols):
            continue
        for col, raw_values in zip(cols, categories):
            cleaned_values = []
            for value in raw_values:
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    cleaned_values.append(text)
            categories_map[col] = sorted(set(cleaned_values))
    return categories_map


def merge_categories(
    generated: dict[str, list[str]], fallback: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged = dict(fallback)
    for key, values in generated.items():
        if values:
            merged[key] = values
    return merged


def default_numeric(feature: str, fallback: float, defaults: dict[str, Any]) -> float:
    value = defaults.get(feature)
    if value is None or pd.isna(value):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def positive_class_index(classes: list[Any]) -> int:
    candidates = {"1", "true", "yes", "positive", "event", "complication", "worsened"}
    for idx, value in enumerate(classes):
        label = str(value).strip().lower()
        if value in (1, True) or label in candidates:
            return idx
    return len(classes) - 1 if classes else 0


def predict_probability(model: Any, X: pd.DataFrame) -> tuple[float, Any]:
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("Invalid predict_proba output format.")

    classes = list(getattr(model, "classes_", []))
    if probabilities.shape[1] == 1:
        class_label = classes[0] if classes else "positive"
        return float(probabilities[0][0]), class_label

    if len(classes) != probabilities.shape[1]:
        classes = [f"class_{i}" for i in range(probabilities.shape[1])]

    idx = positive_class_index(classes)
    return float(probabilities[0][idx]), classes[idx]


st.sidebar.header("Configuration")
models_map = find_models(OUTPUTS_ROOT)
if not models_map:
    st.sidebar.error("No models found in outputs/.")
    st.stop()

targets = sorted(models_map.keys())
target = st.sidebar.selectbox("Target", targets)
target_dir = OUTPUTS_ROOT / target

metadata = load_metadata(str(target_dir / "metadata.json"))
data_config = metadata.get("data_configuration", {})
ui_config = load_json_file(str(APP_FEATURE_CONFIG_FILE))
if ui_config:
    data_config = ui_config

model_names = sorted(models_map[target].keys())
model_name = st.sidebar.selectbox("Model", model_names)
pipeline_path = models_map[target][model_name]
st.sidebar.code(str(pipeline_path))

try:
    model = load_pipeline(str(pipeline_path))
    st.sidebar.success("Pipeline loaded")
except Exception as exc:
    st.sidebar.error("Pipeline loading error")
    st.sidebar.exception(exc)
    st.stop()

model_features, cols_string, cols_date, cols_multi = derive_feature_config(data_config, model)
if not model_features:
    st.error("Unable to derive feature list from metadata or pipeline.")
    st.stop()

features = model_features
if APP_FEATURE_CONFIG_FILE.exists():
    st.sidebar.success(f"UI feature config: {APP_FEATURE_CONFIG_FILE.name}")
else:
    st.sidebar.warning("UI feature config not found; using model metadata.")
st.sidebar.caption(f"Strict mode: showing only model features ({len(features)})")
with st.sidebar.expander("Model features in use"):
    st.write(features)

cols_string_set = set(cols_string)
cols_date_set = set(cols_date)
cols_multi_set = set(cols_multi)

feature_defaults = infer_defaults_from_pipeline(model)
pipeline_categories = infer_categories_from_pipeline(model)
prepared_categories = load_category_options(str(CATEGORY_OPTIONS_FILE))
categories = merge_categories(prepared_categories, pipeline_categories)
if CATEGORY_OPTIONS_FILE.exists():
    st.sidebar.success(f"Loaded category options from {CATEGORY_OPTIONS_FILE.name}")
else:
    st.sidebar.warning(
        "category_options.json not found; using pipeline-derived categories only."
    )

st.subheader("Patient Data Entry")
with st.form("patient_form"):
    values: dict[str, Any] = {}

    selectable_features = []
    numeric_features = []
    for feature in features:
        if (
            feature in cols_string_set
            or feature in cols_date_set
            or feature in cols_multi_set
            or is_symptom_feature(feature)
        ):
            selectable_features.append(feature)
        else:
            numeric_features.append(feature)

    if selectable_features:
        st.markdown("**Selectable / Categorical Fields**")
        sel_col1, sel_col2 = st.columns(2)
        for idx, feature in enumerate(selectable_features):
            col = [sel_col1, sel_col2][idx % 2]
            with col:
                if feature in cols_date_set:
                    values[feature] = st.date_input(
                        feature,
                        value=date.today(),
                        min_value=date(1900, 1, 1),
                        max_value=date(2100, 12, 31),
                        key=f"date_{feature}",
                    )
                elif feature in cols_string_set:
                    options = categories.get(feature, [])
                    if options:
                        values[feature] = st.selectbox(
                            feature,
                            options=[""] + options,
                            index=0,
                            key=f"str_{feature}",
                        )
                    else:
                        values[feature] = st.text_input(
                            feature,
                            value="",
                            key=f"text_{feature}",
                        )
                elif feature in cols_multi_set:
                    values[feature] = st.text_input(
                        f"{feature} (multi)",
                        value="",
                        key=f"multi_{feature}",
                    )
                else:
                    values[feature] = st.checkbox(
                        feature, value=False, key=f"check_{feature}"
                    )

    if numeric_features:
        st.markdown("**Numeric Fields**")
        num_col1, num_col2 = st.columns(2)
        for idx, feature in enumerate(numeric_features):
            col = [num_col1, num_col2][idx % 2]
            with col:
                low_feature = feature.lower()
                if low_feature == "age":
                    default_age = int(round(default_numeric(feature, 60, feature_defaults)))
                    values[feature] = st.number_input(
                        "Age",
                        min_value=0,
                        max_value=120,
                        value=default_age,
                        key=f"num_{feature}",
                    )
                elif "kps" in low_feature:
                    default_kps = int(round(default_numeric(feature, 80, feature_defaults)))
                    default_kps = min(max(default_kps, 0), 100)
                    values[feature] = st.slider(
                        feature,
                        min_value=0,
                        max_value=100,
                        value=default_kps,
                        key=f"slider_{feature}",
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        value=float(default_numeric(feature, 0.0, feature_defaults)),
                        key=f"num_{feature}",
                    )

    submitted = st.form_submit_button("Calculate risk", type="primary")

if submitted:
    row: dict[str, Any] = {}
    for feature in features:
        value = values.get(feature)
        if isinstance(value, bool):
            row[feature] = 1 if value else 0
        elif feature in cols_date_set:
            row[feature] = pd.to_datetime(value)
        elif isinstance(value, str):
            cleaned = value.strip()
            row[feature] = pd.NA if cleaned == "" else cleaned
        else:
            row[feature] = value

    X = pd.DataFrame([row], columns=features)
    st.write("Input sent to model:")
    st.dataframe(X)

    try:
        prediction = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            classes = list(getattr(model, "classes_", []))
            if len(classes) != probabilities.shape[1]:
                classes = [f"class_{i}" for i in range(probabilities.shape[1])]
            proba_row = probabilities[0]

            st.write("Class probabilities:")
            st.dataframe(
                pd.DataFrame(
                    {"class": classes, "probability": proba_row}
                )
            )
            st.caption(f"Probability sum: {float(proba_row.sum()):.4f}")

            probability, positive_label = predict_probability(model, X)
            st.success(f"{target} probability: {probability * 100:.1f}%")
            st.caption(f"Positive class used: {positive_label}")
        st.info(f"Predicted class: {prediction}")
    except Exception as exc:
        st.error("Prediction error.")
        st.exception(exc)
