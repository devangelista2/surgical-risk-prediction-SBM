from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_ROOT = BASE_DIR / "outputs"
DEFAULT_DATASET = BASE_DIR / "data" / "SBM1212.xlsx"
DEFAULT_OUTPUT = OUTPUTS_ROOT / "category_options.json"


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return {}
    return payload


def collect_categorical_columns(outputs_root: Path) -> list[str]:
    found: set[str] = set()
    for metadata_path in outputs_root.rglob("metadata.json"):
        metadata = read_json(metadata_path)
        data_cfg = metadata.get("data_configuration", {})
        if not isinstance(data_cfg, dict):
            continue
        cols = data_cfg.get("cols_string", [])
        if not isinstance(cols, list):
            continue
        for col in cols:
            if isinstance(col, str) and col.strip():
                found.add(col.strip())
    return sorted(found)


def unique_non_empty(values: pd.Series) -> list[str]:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    return sorted(cleaned.unique().tolist())


def build_category_options(df: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for col in columns:
        if col not in df.columns:
            options[col] = []
            continue
        options[col] = unique_non_empty(df[col])
    return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build safe categorical options file for Streamlit inference app."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_DATASET),
        help="Path to private dataset (.xlsx/.xls/.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path (safe artifact used by app.py).",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional explicit list of categorical columns. If omitted, inferred from outputs/*/metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    columns = [c for c in (args.columns or []) if isinstance(c, str) and c.strip()]
    if not columns:
        columns = collect_categorical_columns(OUTPUTS_ROOT.resolve())
    if not columns:
        raise ValueError(
            "No categorical columns provided or found in outputs metadata. "
            "Use --columns to pass them explicitly."
        )

    df = load_table(input_path)
    options = build_category_options(df, columns)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_name": input_path.name,
        "columns": options,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    non_empty_cols = sum(1 for values in options.values() if values)
    print(f"Wrote {output_path}")
    print(f"Columns total: {len(options)}")
    print(f"Columns with values: {non_empty_cols}")


if __name__ == "__main__":
    main()
