from __future__ import annotations

from typing import Any

import pandas as pd


def comparison_frame(recipes: list[dict[str, Any]], numeric_features: list[str]) -> pd.DataFrame:
    rows = []
    for index, recipe in enumerate(recipes, start=1):
        rows.append({"配方": f"配方 {index}", **recipe["record"], "预测屈服强度 (MPa)": float(recipe["prediction"])})
    columns = ["配方", *numeric_features, "Processing", "预测屈服强度 (MPa)"]
    return pd.DataFrame(rows, columns=columns)


def comparison_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def comparison_markdown(frame: pd.DataFrame) -> bytes:
    columns = [str(column) for column in frame.columns]
    lines = ["# 配方对比报告", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")
