from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Yield Strength (MPa)"
PROCESSING = "Processing"
EXCLUDED = {"Elongation (%)", "Tensile Strength (MPa)", "class", TARGET}
ALIASES = {
    "strain harderned (hard)": "Strain hardened (Hard)",
    "strain hardened (hard)": "Strain hardened (Hard)",
    "strain hardened": "Strain hardened",
}


def normalize_processing(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*\+\s*", " + ", text)
    return ALIASES.get(text.casefold(), text)


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "al_data.csv")
    if len(data.columns):
        first = str(data.columns[0])
        values = pd.to_numeric(data.iloc[:, 0], errors="coerce")
        sequential = bool(values.notna().all() and np.array_equal(values.to_numpy(), np.arange(len(data))))
        if first.startswith("Unnamed") or first == "" or first.lower() in {"index", "row_index"} or sequential:
            data = data.iloc[:, 1:].copy()
    numeric_features = [
        col for col in data.columns
        if col not in EXCLUDED and col != PROCESSING and not col.startswith("Unnamed")
    ]
    for name in numeric_features + [TARGET]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data[PROCESSING] = data[PROCESSING].map(normalize_processing).replace("", pd.NA)
    required = numeric_features + [PROCESSING, TARGET]
    data = data.drop_duplicates().dropna(subset=required).reset_index(drop=True)
    # Group keys must use the source proportions, exactly as src/train.py does,
    # before the display/model-unit conversion to percentages.
    groups = data[numeric_features].round(10).astype(str).agg("|".join, axis=1)
    train_index, _ = next(
        GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=2024).split(data, data[TARGET], groups=groups)
    )
    train_data = data.iloc[train_index].copy()
    train_data[numeric_features] = train_data[numeric_features] * 100.0
    output = {
        "source": "data/al_data.csv",
        "cleaning": "drop index, normalize Processing, drop exact duplicates and invalid required rows, multiply source proportions by 100",
        "cleaned_rows": int(len(data)),
        "numeric_features": numeric_features,
        "split": "GroupShuffleSplit(train_size=0.8, random_state=2024) by exact composition combination",
        "training_rows": int(len(train_data)),
        "ranges": {
            name: {
                "q05_percent": float(train_data[name].quantile(0.05)),
                "q95_percent": float(train_data[name].quantile(0.95)),
                "unit": "%",
            }
            for name in numeric_features
        },
    }
    target = ROOT / "artifacts" / "sensitivity_ranges.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
