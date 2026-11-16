from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def build_compensated_scan(
    base_values: Mapping[str, Any],
    processing: str,
    element: str,
    low: float,
    high: float,
    points: int = 41,
) -> pd.DataFrame:
    """Build a single batch while Al offsets every change in the selected element."""
    if element == "Al":
        raise ValueError("Al 是补偿字段，请选择其他元素进行扫描")
    if points < 30 or points > 50:
        raise ValueError("扫描点数必须在 30–50 之间")
    base = {name: float(value) for name, value in base_values.items()}
    if "Al" not in base or element not in base:
        raise ValueError("扫描需要 Al 和所选元素的成分字段")
    base["Al"] += 100.0 - float(sum(base.values()))
    if not 0.0 <= base["Al"] <= 100.0:
        raise ValueError("当前输入无法通过 Al 补偿到 100%，请先调整配方")
    values = np.linspace(float(low), float(high), int(points))
    original_element = base[element]
    rows = []
    for value in values:
        row = dict(base)
        row[element] = float(value)
        row["Al"] = float(base["Al"] - (value - original_element))
        if not 0.0 <= row["Al"] <= 100.0:
            raise ValueError("扫描过程中 Al 超出 0–100%，请调整当前配方")
        row["Processing"] = processing
        rows.append(row)
    frame = pd.DataFrame(rows)
    totals = frame.drop(columns=["Processing"]).sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(totals, 100.0, atol=1e-9):
        raise AssertionError("补偿扫描成分合计未保持 100%")
    return frame
