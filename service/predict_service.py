from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import Booster

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "feature_config.json"
REGISTRY_PATH = ROOT / "model_registry.json"
DB_PATH = ROOT / "records.db"


def _normalize_processing(value: Any) -> str:
    import re
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"\s*\+\s*", " + ", text)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _feature_frame(record_or_df: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    raw = pd.DataFrame([record_or_df]) if isinstance(record_or_df, dict) else record_or_df.copy()
    required = cfg["numeric_features"] + [cfg["processing_field"]]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError("缺少必要字段: " + ", ".join(missing))
    raw[cfg["processing_field"]] = raw[cfg["processing_field"]].map(_normalize_processing)
    for name in cfg["numeric_features"]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    if raw[required].isna().any().any():
        raise ValueError("输入字段包含空值或非数值成分")
    scale = float(cfg.get("source_to_model_scale_factor", 1.0))
    for name in cfg["numeric_features"]:
        raw[name] = raw[name] * scale
    matrix = pd.get_dummies(raw[required], columns=[cfg["processing_field"]], dtype=float)
    matrix = matrix.reindex(columns=cfg["all_model_features"], fill_value=0.0)
    return matrix[cfg["selected_features"]].astype(float)


def _load_model(name: str = "lightgbm"):
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    key = name.lower().replace(" ", "_")
    if key in {"randomforest", "random_forest", "rf"}:
        key = "random_forest"
    elif key in {"xgb", "xgboost"}:
        key = "xgboost"
    entry = registry["models"].get(key, registry["models"]["lightgbm"])
    model_path = ROOT / entry["path"]
    if entry["type"] == "lightgbm":
        return Booster(model_str=model_path.read_text(encoding="utf-8"))
    return joblib.load(model_path)


def predict_dataframe(data: pd.DataFrame, model_name: str = "lightgbm") -> pd.DataFrame:
    features = _feature_frame(data)
    model = _load_model(model_name)
    result = data.copy()
    result["predicted_yield_strength_MPa"] = np.asarray(model.predict(features), dtype=float)
    return result


def predict_record(record: dict[str, Any], model_name: str = "lightgbm", save: bool = True) -> dict[str, Any]:
    result_df = predict_dataframe(pd.DataFrame([record]), model_name)
    prediction = float(result_df.iloc[0]["predicted_yield_strength_MPa"])
    result = {
        "id": str(uuid.uuid4()), "model": model_name,
        "predicted_yield_strength_MPa": prediction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {key: value for key, value in record.items()},
    }
    if save:
        save_record(result)
    return result


def explain_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return per-feature TreeSHAP values for one prediction."""
    import shap
    features = _feature_frame(record)
    model = _load_model("lightgbm")
    values = shap.TreeExplainer(model).shap_values(features)
    if isinstance(values, list):
        values = values[0]
    contributions = np.asarray(values, dtype=float).reshape(-1)
    prediction = float(model.predict(features)[0])
    return {
        "base_value_MPa": prediction - float(contributions.sum()),
        "prediction_MPa": prediction,
        "contributions": [
            {"feature": name, "shap_value_MPa": float(value)}
            for name, value in sorted(zip(features.columns, contributions), key=lambda pair: abs(pair[1]), reverse=True)
        ],
    }


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS prediction_records (id TEXT PRIMARY KEY, created_at TEXT, model TEXT, prediction_mpa REAL, input_json TEXT NOT NULL)"
        )


def save_record(result: dict[str, Any]) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO prediction_records VALUES (?, ?, ?, ?, ?)",
            (result["id"], result["created_at"], result["model"], result["predicted_yield_strength_MPa"], json.dumps(result["input"], ensure_ascii=False)),
        )


def list_records(model: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT id, created_at, model, prediction_mpa, input_json FROM prediction_records"
    parameters: list[Any] = []
    if model:
        query += " WHERE model = ?"
        parameters.append(model)
    query += " ORDER BY created_at DESC LIMIT ?"
    parameters.append(int(limit))
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [
        {"id": row[0], "created_at": row[1], "model": row[2], "predicted_yield_strength_MPa": row[3], "input": json.loads(row[4])}
        for row in rows
    ]


def clear_records() -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM prediction_records")
