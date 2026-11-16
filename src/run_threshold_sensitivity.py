from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train
from theme import PRIMARY, PURPLE, TEAL


def prepare_data() -> tuple[list[str], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
    metrics_doc = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / "data" / "al_data.csv")
    first = str(raw.columns[0]) if len(raw.columns) else ""
    first_values = pd.to_numeric(raw.iloc[:, 0], errors="coerce") if len(raw.columns) else pd.Series(dtype=float)
    sequential_index = bool(len(raw) and first_values.notna().all()
                            and np.array_equal(first_values.to_numpy(), np.arange(len(raw))))
    if first.startswith("Unnamed") or first == "" or first.lower() in {"index", "row_index"} or sequential_index:
        raw = raw.iloc[:, 1:].copy()
    numeric = [name for name in raw.columns if name not in train.EXCLUDED and name != train.PROCESSING
               and not str(name).startswith("Unnamed")]
    for name in numeric:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    raw[train.TARGET] = pd.to_numeric(raw[train.TARGET], errors="coerce")
    raw[train.PROCESSING] = raw[train.PROCESSING].map(train.normalize_processing).replace("", np.nan)
    raw = raw.drop_duplicates().reset_index(drop=True)
    raw = raw.dropna(subset=numeric + [train.PROCESSING, train.TARGET]).reset_index(drop=True)
    y = raw[train.TARGET].astype(float)
    groups = raw[numeric].round(10).astype(str).agg("|".join, axis=1)
    seed = int(metrics_doc["seed"])
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
                               .split(raw, y, groups=groups))
    if set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx])):
        raise AssertionError("Composition groups overlap between train and test")
    x_train, feature_order = train.normalize_and_matrix(raw.iloc[train_idx], numeric)
    x_test, _ = train.normalize_and_matrix(raw.iloc[test_idx], numeric)
    x_test = x_test.reindex(columns=feature_order, fill_value=0.0)
    if len(train_idx) != metrics_doc["train_samples"] or len(test_idx) != metrics_doc["test_samples"]:
        raise AssertionError("Rebuilt split sample counts differ from metrics.json")
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return numeric, x_train, y_train, x_test, y_test, str(metrics_doc["seed"])


def main() -> None:
    metric_path = ROOT / "metrics.json"
    metrics_before = json.loads(metric_path.read_text(encoding="utf-8"))
    original_metrics_text = metric_path.read_bytes()
    _, x_train, y_train, x_test, y_test, seed_text = prepare_data()
    seed = int(seed_text)
    params = metrics_before["tuning"]["parameters"]
    _, gain_report, selection_info = train.select_features(x_train, y_train, seed)
    gain_order = gain_report["feature"].tolist()
    rows = []
    for ratio in (0.50, 0.65, 0.80):
        count = min(max(8, int(math.ceil(len(gain_order) * ratio))), len(gain_order))
        selected = gain_order[:count]
        model = train.fit_lgbm(x_train[selected], y_train, params, seed)
        predictions = model.predict(x_test[selected])
        rows.append({"gain_retention_ratio": ratio, "feature_count": len(selected),
                     **train.metrics(y_test.to_numpy(), predictions),
                     "train_samples": len(y_train), "test_samples": len(y_test), "seed": seed})
    result = pd.DataFrame(rows)
    csv_path = ROOT / "evidence" / "特征选择阈值敏感性.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.12g")

    train.configure_plot_font()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    specifications = [("MAE_MPa", "MAE (MPa)"), ("RMSE_MPa", "RMSE (MPa)"), ("R2", "R²（无量纲）")]
    colors = [PRIMARY, PURPLE, TEAL]
    for ax, (metric, label) in zip(axes, specifications):
        ax.bar([f"{value:.0%}" for value in result["gain_retention_ratio"]], result[metric], color=colors)
        ax.set(title=label, xlabel="Gain 特征保留比例", ylabel=label)
    fig.suptitle("Gain 特征保留比例敏感性（同一测试划分）")
    fig.tight_layout()
    figure_path = ROOT / "artifacts" / "figures" / "09_threshold_sensitivity.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    lines = [
        "# 特征选择阈值敏感性说明", "",
        f"本附加实验复用主评估的 seed {seed}、按成分组合分组的训练/测试划分（{len(y_train)}/{len(y_test)}）。近零方差、相关性去冗余和 LightGBM gain 排序只在训练集执行一次，并沿用项目原有特征选择器配置；三档比例从同一 gain 排序列表截取。每档最终评估模型均采用 `metrics.json` 中的调优后参数并各训练一次；留出测试集仅计算该档结果。", "",
        "| Gain 保留比例 | 最终特征数 | MAE (MPa) | RMSE (MPa) | R² |", "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['gain_retention_ratio']:.0%} | {row['feature_count']} | {row['MAE_MPa']:.6f} | {row['RMSE_MPa']:.6f} | {row['R2']:.6f} |")
    lines += [
        "", "项目采用 65% 是已登记的主特征选择规则，本附加实验用于展示该规则对阈值变化的敏感程度，并不根据这三次测试结果重新挑选主模型阈值。65% 在本数据上保留中等数量的特征，并延续已训练模型、`feature_config.json`、`selection_report.csv` 与网页输入特征之间的一致口径；该档结果应与正式的 65% 评估保持一致。正式指标与模型均保持不变。三档误差差异只适用于当前留出划分，不代表外部泛化结论。", "",
        f"机器生成结果：`{csv_path.relative_to(ROOT).as_posix()}`；对比图：`{figure_path.relative_to(ROOT).as_posix()}`。",
    ]
    (ROOT / "evidence" / "特征选择阈值敏感性说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metric_path.read_bytes() != original_metrics_text:
        raise AssertionError("metrics.json was modified by the sensitivity experiment")
    metrics_after = json.loads(metric_path.read_text(encoding="utf-8"))
    if metrics_after["models"] != metrics_before["models"] or metrics_after["stability"] != metrics_before["stability"]:
        raise AssertionError("Official model metrics or stability results changed")
    print(result.to_string(index=False))
    print(f"Wrote {csv_path.relative_to(ROOT)} and {figure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
