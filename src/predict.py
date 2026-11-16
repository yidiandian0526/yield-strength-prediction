"""Run the trained yield-strength model on an input CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.predict_service import load_config, predict_dataframe  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用已训练模型预测 CSV 中每条铝合金配方的屈服强度。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入 CSV 路径；元素成分应为 0–100 百分比。",
    )
    parser.add_argument(
        "--model",
        default="lightgbm",
        help="模型名称（默认 lightgbm；也可使用 random_forest 或 xgboost）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 CSV 路径；默认在输入文件旁生成 <文件名>_predictions.csv。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser()
    output_path = args.output.expanduser() if args.output else input_path.with_name(
        f"{input_path.stem}_predictions.csv"
    )

    try:
        data = pd.read_csv(input_path, encoding="utf-8-sig")
        if data.empty:
            raise ValueError("输入 CSV 没有数据行")

        # Field names, units and processing categories are owned by the same
        # feature_config.json used by the web app and REST API.
        config = load_config()
        required = config["numeric_features"] + [config["processing_field"]]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError("缺少必要字段: " + ", ".join(missing))

        predictions = predict_dataframe(data, model_name=args.model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False, encoding="utf-8-sig")

        first_value = float(predictions.iloc[0]["predicted_yield_strength_MPa"])
        # Keep the CLI summary ASCII so it remains readable in legacy Windows
        # console code pages while retaining the requested row count/value.
        print(f"prediction_rows: {len(predictions)}")
        print(f"first_prediction_MPa: {first_value:.6f}")
        print(f"output_file: {output_path}")
        return 0
    except Exception as exc:
        print(f"预测失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

