from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
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
    text = re.sub(r"\s*\+\s*", " + ", text)
    aliases = load_config().get("processing_aliases", {})
    return aliases.get(text.casefold(), text)


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
    numeric = raw[cfg["numeric_features"]]
    out_of_range = (numeric.lt(0.0) | numeric.gt(100.0))
    if out_of_range.any().any():
        details = [
            f"{column}={value:g}"
            for column in numeric.columns
            for value in numeric.loc[out_of_range[column], column].head(5)
        ]
        raise ValueError("元素成分必须在 0–100% 范围内；超范围字段：" + ", ".join(details))

    processing_field = cfg["processing_field"]
    allowed = cfg["processing_values"]
    unknown = sorted({value for value in raw[processing_field].astype(str) if value not in allowed})
    if unknown:
        raise ValueError(
            "未知 Processing 值："
            + ", ".join(repr(value) for value in unknown)
            + "。允许值："
            + "、".join(allowed)
        )

    composition_sum = numeric.sum(axis=1)
    warnings = []
    for row_position, total in enumerate(composition_sum, start=1):
        if abs(float(total) - 100.0) > 1.0:
            warnings.append(
                f"第 {row_position} 行元素成分合计为 {total:.3f}%，与 100% 相差超过 1 个百分点；预测已继续，请核对输入。"
            )
    # Service inputs are already model-space percentages (0–100).  The
    # source CSV conversion from proportions happens only in train.py;
    # applying that factor here would scale user input a second time.
    scale = float(cfg.get("service_input_scale_factor", 1.0))
    if scale != 1.0:
        raise ValueError("service_input_scale_factor 必须为 1.0；服务层输入应为模型百分比")
    matrix = pd.get_dummies(raw[required], columns=[cfg["processing_field"]], dtype=float)
    unexpected_columns = [name for name in matrix.columns if name not in cfg["all_model_features"]]
    if unexpected_columns:
        raise ValueError(
            "输入生成了模型未见过的特征："
            + ", ".join(unexpected_columns)
            + "。请核对 Processing 是否为允许的枚举值。"
        )
    matrix = matrix.reindex(columns=cfg["all_model_features"], fill_value=0.0)
    matrix = matrix[cfg["selected_features"]].astype(float)
    matrix.attrs["input_warnings"] = warnings
    return matrix


@lru_cache(maxsize=3)
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


def load_model(name: str = "lightgbm"):
    return _load_model(name)


def predict_dataframe(data: pd.DataFrame, model_name: str = "lightgbm", model=None) -> pd.DataFrame:
    features = _feature_frame(data)
    model = model if model is not None else _load_model(model_name)
    result = data.copy()
    result["predicted_yield_strength_MPa"] = np.asarray(model.predict(features), dtype=float)
    result.attrs["input_warnings"] = features.attrs.get("input_warnings", [])
    return result


def predict_record(record: dict[str, Any], model_name: str = "lightgbm", save: bool = True, model=None) -> dict[str, Any]:
    result_df = predict_dataframe(pd.DataFrame([record]), model_name, model=model)
    prediction = float(result_df.iloc[0]["predicted_yield_strength_MPa"])
    result = {
        "id": str(uuid.uuid4()), "model": model_name,
        "predicted_yield_strength_MPa": prediction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {key: value for key, value in record.items()},
        "warnings": result_df.attrs.get("input_warnings", []),
    }
    if save:
        save_record(result)
    return result


def explain_record(record: dict[str, Any], model=None) -> dict[str, Any]:
    """Return per-feature TreeSHAP values for one prediction."""
    import shap
    features = _feature_frame(record)
    model = model if model is not None else _load_model("lightgbm")
    values = shap.TreeExplainer(model).shap_values(features)
    if isinstance(values, list):
        values = values[0]
    contributions = np.asarray(values, dtype=float).reshape(-1)
    prediction = float(model.predict(features)[0])
    return {
        "base_value_MPa": prediction - float(contributions.sum()),
        "prediction_MPa": prediction,
        "warnings": features.attrs.get("input_warnings", []),
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
