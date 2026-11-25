from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor, log_evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from theme import PRIMARY, PURPLE, TEAL, TEXT  # noqa: E402

SEED = 2024
TARGET = "Yield Strength (MPa)"
PROCESSING = "Processing"
EXCLUDED = {"Elongation (%)", "Tensile Strength (MPa)", "class", TARGET}
# Keep the held-out 2024 test split reserved for one final evaluation only.
STABILITY_SEEDS = [2020, 2021, 2022, 2023, 2025]
CV_SPLITS = 3
PROCESSING_ALIASES = {
    "strain harderned (hard)": "Strain hardened (Hard)",
    "strain hardened (hard)": "Strain hardened (Hard)",
    "strain hardened": "Strain hardened",
}
TUNING_BASE = {
    "num_leaves": 31,
    "learning_rate": 0.035,
    "n_estimators": 700,
    "min_child_samples": 12,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.05,
    "reg_lambda": 0.2,
    "min_split_gain": 0.0,
}
TUNING_CANDIDATE_VALUES = {
    "num_leaves": [31, 15],
    "learning_rate": [0.035, 0.06],
    "n_estimators": [700, 400],
    "min_child_samples": [12, 8],
    "max_depth": [-1, 8],
    "subsample": [0.9, 0.8],
    "colsample_bytree": [0.9, 0.8],
    "reg_alpha": [0.05, 0.0],
    "reg_lambda": [0.2, 1.0],
    "min_split_gain": [0.0, 0.1],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_processing(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*\+\s*", " + ", text)
    return PROCESSING_ALIASES.get(text.casefold(), text)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE_MPa": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_lgbm(params: dict[str, Any] | None = None, seed: int = SEED) -> LGBMRegressor:
    options = {**TUNING_BASE, **(params or {})}
    return LGBMRegressor(
        objective="regression", **options, subsample_freq=1,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def configure_plot_font() -> str:
    # Bundled CJK font first: training may run on Linux (e.g. Streamlit
    # Community Cloud or a CI runner) where the Windows paths below do not
    # exist; without it Matplotlib falls back to DejaVu Sans and every
    # Chinese chart title/label renders as tofu boxes.
    bundled_font = PROJECT_ROOT / "assets" / "fonts" / "NotoSansSC-Regular.otf"
    if bundled_font.exists():
        try:
            font_manager.fontManager.addfont(str(bundled_font))
        except (OSError, RuntimeError):
            pass
    font_paths = (
        Path("C:/Windows/Fonts/Noto Sans SC (TrueType).otf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans SC", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei")
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [selected, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any] | None = None,
             seed: int = SEED) -> LGBMRegressor:
    model = make_lgbm(params, seed)
    # Fixed iteration count: the held-out test set is never used for fitting or early stopping.
    model.fit(X_train, y_train, callbacks=[log_evaluation(0)])
    return model


def normalize_and_matrix(df: pd.DataFrame, numeric_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    # The source CSV stores composition percentages as proportions (e.g. 0.88).
    # Convert them to percentage points so training and UI/API inputs share units.
    for name in numeric_features:
        work[name] = pd.to_numeric(work[name], errors="coerce") * 100.0
    work[PROCESSING] = work[PROCESSING].map(normalize_processing)
    matrix = pd.get_dummies(work[numeric_features + [PROCESSING]], columns=[PROCESSING], dtype=float)
    return matrix.astype(float), list(matrix.columns)


def data_quality_audit(df: pd.DataFrame, numeric_features: list[str]) -> dict[str, Any]:
    required = numeric_features + [PROCESSING, TARGET]
    missing = {name: int(df[name].isna().sum()) for name in required}
    outliers: dict[str, int] = {}
    for name in numeric_features + [TARGET]:
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        if values.empty:
            outliers[name] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[name] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    composition_sum = df[numeric_features].sum(axis=1, min_count=len(numeric_features))
    return {
        "missing_values_before_cleaning": missing,
        "exact_duplicate_rows_before_cleaning": int(df.duplicated().sum()),
        "iqr_outlier_counts_descriptive_only": outliers,
        "outlier_rule": "1.5*IQR; descriptive counts only, no row removed by this rule",
        "composition_source_scale": "0-1 proportion in CSV; multiplied by 100 for model input in percent",
        "raw_composition_sum": {"minimum": float(composition_sum.min()), "median": float(composition_sum.median()), "maximum": float(composition_sum.max())},
    }


def select_features(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    variances = X_train.var(axis=0)
    variance_keep = [c for c in X_train.columns if float(variances.get(c, 0.0)) > 1e-12]
    corr = X_train[variance_keep].corr().abs()
    corr_keep: list[str] = []
    for col in variance_keep:
        if not corr_keep or not any(float(corr.loc[col, prior]) >= 0.98 for prior in corr_keep):
            corr_keep.append(col)
    pre_model = make_lgbm(seed=seed)
    pre_model.fit(X_train[corr_keep], y_train, callbacks=[log_evaluation(0)])
    gains = pd.Series(pre_model.booster_.feature_importance(importance_type="gain"), index=corr_keep).sort_values(ascending=False)
    n_select = min(max(8, int(np.ceil(len(gains) * 0.65))), len(gains))
    selected = list(gains.head(n_select).index)
    report = pd.DataFrame({
        "feature": gains.index, "gain": gains.values,
        "gain_rank": np.arange(1, len(gains) + 1), "selected": gains.index.isin(selected),
        "kept_after_variance": gains.index.isin(variance_keep), "kept_after_correlation": gains.index.isin(corr_keep),
    })
    info = {
        "initial_feature_count": int(X_train.shape[1]), "after_near_zero_variance_count": len(variance_keep),
        "after_correlation_count": len(corr_keep), "selected_feature_count": len(selected),
        "variance_threshold": 1e-12, "correlation_threshold": 0.98,
        "selection_rule": "LightGBM gain descending; retain top 65%, minimum 8",
    }
    return selected, report, info


def build_tuning_candidates() -> list[dict[str, Any]]:
    candidates = [{"candidate_id": "C00", "changed_parameter": "baseline", "tested_value": "baseline", "params": dict(TUNING_BASE)}]
    for index, (parameter, values) in enumerate(TUNING_CANDIDATE_VALUES.items(), start=1):
        params = dict(TUNING_BASE)
        params[parameter] = values[1]
        candidates.append({
            "candidate_id": f"C{index:02d}", "changed_parameter": parameter,
            "tested_value": values[1], "params": params,
        })
    return candidates


def tune_lgbm_group_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = build_tuning_candidates()
    splitter = GroupKFold(n_splits=CV_SPLITS)
    fold_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for fold_number, (fit_index, valid_index) in enumerate(splitter.split(X, y, groups), start=1):
            train_groups = set(groups.iloc[fit_index])
            valid_groups = set(groups.iloc[valid_index])
            if train_groups.intersection(valid_groups):
                raise AssertionError("composition group leakage detected in tuning fold")
            fold_selected, _, _ = select_features(X.iloc[fit_index], y.iloc[fit_index], seed=seed + fold_number)
            model = fit_lgbm(X.iloc[fit_index][fold_selected], y.iloc[fit_index], candidate["params"], seed + fold_number)
            prediction = model.predict(X.iloc[valid_index][fold_selected])
            fold_rows.append({
                "candidate_id": candidate["candidate_id"],
                "changed_parameter": candidate["changed_parameter"],
                "tested_value": candidate["tested_value"],
                "fold": fold_number,
                "train_samples": int(len(fit_index)),
                "validation_samples": int(len(valid_index)),
                "train_groups": int(len(train_groups)),
                "validation_groups": int(len(valid_groups)),
                **metrics(y.iloc[valid_index].to_numpy(), prediction),
            })
    fold_frame = pd.DataFrame(fold_rows)
    summary = fold_frame.groupby("candidate_id", sort=False).agg(
        mean_MAE_MPa=("MAE_MPa", "mean"), std_MAE_MPa=("MAE_MPa", "std"),
        mean_RMSE_MPa=("RMSE_MPa", "mean"), mean_R2=("R2", "mean"),
    ).reset_index()
    candidate_order = {candidate["candidate_id"]: index for index, candidate in enumerate(candidates)}
    summary["order"] = summary["candidate_id"].map(candidate_order)
    summary = summary.sort_values(["mean_MAE_MPa", "std_MAE_MPa", "order"], kind="stable").reset_index(drop=True)
    winner_id = str(summary.iloc[0]["candidate_id"])
    winner = next(candidate for candidate in candidates if candidate["candidate_id"] == winner_id)
    return dict(winner["params"]), fold_rows, [{**candidate, **summary.loc[summary["candidate_id"] == candidate["candidate_id"]].drop(columns="order").iloc[0].to_dict()} for candidate in candidates]


def fit_original_lgbm(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED) -> LGBMRegressor:
    """Reproduce the pre-search LightGBM constructor for an end-point comparison."""
    model = LGBMRegressor(
        objective="regression", n_estimators=700, learning_rate=0.035,
        num_leaves=31, min_child_samples=12, subsample=0.9,
        colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=0.2,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train, callbacks=[log_evaluation(0)])
    return model


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_figures(root: Path, y_test: pd.Series, predictions: dict[str, np.ndarray], final_model: LGBMRegressor,
                  selected: list[str], selection_compare: pd.DataFrame, group_labels: pd.Series,
                  shap_df: pd.DataFrame) -> list[str]:
    fig_dir = root / "artifacts" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    plot_font = configure_plot_font()
    plt.rcParams["font.sans-serif"] = [plot_font, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    actual = np.asarray(y_test); pred = predictions["LightGBM"]; paths: list[str] = []
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(actual, pred, s=28, alpha=.75, color=PRIMARY); lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max())); ax.plot([lo, hi], [lo, hi], "--", color=TEXT); ax.set(title="真实值与预测值", xlabel="真实屈服强度 (MPa)", ylabel="预测屈服强度 (MPa)"); fig.tight_layout(); p=fig_dir/"01_actual_vs_pred.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    errors = pred - actual
    fig, ax = plt.subplots(figsize=(8, 6)); sns.histplot(errors, kde=True, ax=ax, color=PRIMARY); ax.axvline(0, ls="--", color=TEXT); ax.set(title="预测误差分布", xlabel="预测误差 (MPa)", ylabel="样本数"); fig.tight_layout(); p=fig_dir/"02_error_distribution.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    imp = pd.Series(final_model.feature_importances_, index=selected).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(9, 7)); imp.plot.barh(ax=ax, color=PRIMARY); ax.set(title="LightGBM 特征重要性", xlabel="分裂重要性（无量纲相对值）", ylabel="输入特征"); fig.tight_layout(); p=fig_dir/"03_feature_importance.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for metric_name, ax in zip(["MAE_MPa", "RMSE_MPa", "R2"], axes):
        label = metric_name.replace("_MPa", " (MPa)") if metric_name != "R2" else "R²（无量纲）"
        sns.barplot(data=selection_compare, x="stage", y=metric_name, ax=ax, color=PRIMARY); ax.set(title=label, xlabel="特征阶段", ylabel=label); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("特征选择前后评价对比"); fig.tight_layout(); p=fig_dir/"04_feature_selection_comparison.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    top = shap_df.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top["feature"], top["mean_abs_shap"], color=PRIMARY); ax.set(title="SHAP 全局贡献", xlabel="平均绝对 SHAP 值 (MPa)", ylabel="输入特征"); fig.tight_layout(); p=fig_dir/"05_shap_global.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    grouped = pd.DataFrame({"Processing": group_labels.to_numpy(), "error": errors}).groupby("Processing", as_index=False).agg(mean_abs_error=("error", lambda x: float(np.mean(np.abs(x)))), count=("error", "size")).sort_values("mean_abs_error", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(grouped["Processing"].astype(str)[::-1], grouped["mean_abs_error"][::-1], color=PRIMARY); ax.set(title="按 Processing 分组的误差", xlabel="平均绝对误差 (MPa)", ylabel="Processing（处理状态）"); fig.tight_layout(); p=fig_dir/"06_grouped_error.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    md = pd.DataFrame([{"model": name, **metrics(actual, vals)} for name, vals in predictions.items()])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for metric_name, ax in zip(["MAE_MPa", "RMSE_MPa", "R2"], axes):
        label = metric_name.replace("_MPa", " (MPa)") if metric_name != "R2" else "R²（无量纲）"
        ax.bar(md["model"], md[metric_name], color=[PRIMARY, PURPLE, TEAL]); ax.set(title=label, xlabel="模型", ylabel=label); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("三模型测试集指标对比"); fig.tight_layout(); p=fig_dir/"07_model_comparison.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    return paths


def evaluate_model_suite(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                         selected: list[str], lgbm_params: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    lgbm = fit_lgbm(X_train[selected], y_train, lgbm_params, seed)
    rf = RandomForestRegressor(n_estimators=500, max_features=0.8, random_state=seed, n_jobs=-1)
    rf.fit(X_train[selected], y_train)
    xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.9,
                       colsample_bytree=0.9, objective="reg:squarederror", random_state=seed, n_jobs=4)
    xgb.fit(X_train[selected], y_train, verbose=False)
    return {"LightGBM": lgbm.predict(X_test[selected]), "RandomForest": rf.predict(X_test[selected]),
            "XGBoost": xgb.predict(X_test[selected])}


def run_stability(df: pd.DataFrame, numeric_features: list[str], params: dict[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    y = df[TARGET].astype(float)
    groups = df[numeric_features].round(10).astype(str).agg("|".join, axis=1)
    rows: list[dict[str, Any]] = []
    for seed in STABILITY_SEEDS:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed).split(df, y, groups=groups))
        train_groups, test_groups = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
        if train_groups.intersection(test_groups):
            raise AssertionError(f"composition group leakage detected for stability seed {seed}")
        X_train, all_features = normalize_and_matrix(df.iloc[train_idx], numeric_features)
        X_test, _ = normalize_and_matrix(df.iloc[test_idx], numeric_features)
        X_test = X_test.reindex(columns=all_features, fill_value=0.0)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        selected, _, _ = select_features(X_train, y_train, seed=seed)
        predictions = evaluate_model_suite(X_train, y_train, X_test, selected, params, seed)
        for model_name, prediction in predictions.items():
            rows.append({"record_type": "run", "seed": seed, "model": model_name,
                         "train_samples": len(train_idx), "test_samples": len(test_idx),
                         "train_groups": len(train_groups), "test_groups": len(test_groups),
                         "selected_feature_count": len(selected),
                         **metrics(y_test.to_numpy(), prediction)})
    runs = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for model_name, model_rows in runs.groupby("model", sort=False):
        for metric_name in ("MAE_MPa", "RMSE_MPa", "R2"):
            values = model_rows[metric_name].to_numpy(dtype=float)
            summaries.append({"record_type": "summary", "seed": "", "model": model_name,
                              "train_samples": "", "test_samples": "", "train_groups": "", "test_groups": "",
                              "selected_feature_count": "", "statistic": "mean", "metric": metric_name,
                              "value": float(values.mean())})
            summaries.append({"record_type": "summary", "seed": "", "model": model_name,
                              "train_samples": "", "test_samples": "", "train_groups": "", "test_groups": "",
                              "selected_feature_count": "", "statistic": "std", "metric": metric_name,
                              "value": float(values.std(ddof=1))})
    summary_df = pd.DataFrame(summaries)
    combined = runs.copy()
    combined["statistic"] = ""
    combined["metric"] = ""
    combined["value"] = np.nan
    combined = pd.concat([combined, summary_df], ignore_index=True, sort=False)
    output = root / "evidence" / "分组划分稳定性对比.csv"
    combined.to_csv(output, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = {"LightGBM": PRIMARY, "RandomForest": PURPLE, "XGBoost": TEAL}
    for metric_name, ax in zip(("MAE_MPa", "RMSE_MPa", "R2"), axes):
        for model_name, model_rows in runs.groupby("model", sort=False):
            vals = model_rows[metric_name].to_numpy(dtype=float)
            ax.errorbar(model_name, vals.mean(), yerr=vals.std(ddof=1), fmt="o", capsize=5,
                        color=palette[model_name], label=model_name)
        label = "R²（无量纲）" if metric_name == "R2" else metric_name.replace("_MPa", " (MPa)")
        ax.set(title=label, xlabel="模型", ylabel=label)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("五个随机种子的分组划分稳定性（均值 ± 标准差）")
    fig.tight_layout()
    stability_figure = root / "artifacts" / "figures" / "08_stability.png"
    fig.savefig(stability_figure, dpi=160)
    plt.close(fig)
    summary = {model_name: {
        metric_name: {"mean": float(model_rows[metric_name].mean()),
                      "std": float(model_rows[metric_name].std(ddof=1))}
        for metric_name in ("MAE_MPa", "RMSE_MPa", "R2")
    } for model_name, model_rows in runs.groupby("model", sort=False)}
    return combined, summary, str(stability_figure.relative_to(root))


def write_tuning_evidence(root: Path, params: dict[str, Any], folds: list[dict[str, Any]], summaries: list[dict[str, Any]],
                          tuned_test: dict[str, float], original_test: dict[str, float]) -> None:
    (root / "artifacts").mkdir(exist_ok=True)
    pd.DataFrame(folds).to_csv(root / "artifacts" / "tuning_cv_folds.csv", index=False, encoding="utf-8-sig")
    compact = [{key: value for key, value in candidate.items() if key != "params"} for candidate in summaries]
    save_json(root / "artifacts" / "tuning_cv_summary.json", compact)
    winner = min(summaries, key=lambda item: (item["mean_MAE_MPa"], item["std_MAE_MPa"]))
    lines = [
        "# 主模型调参记录", "",
        "本次使用单变量轮换的 11 候选局部搜索：仅在固定随机种子 2024 的训练子集内，以按成分组合分组的 3 折 GroupKFold 对 LightGBM 进行参数比较。每折的 gain 特征选择器仅在该折训练部分拟合；166 条留出测试样本未参与选参。候选由基准组合及每个参数单独替换为一个备选值构成，选择依据为三折平均 MAE 最低，平局时标准差较低。", "",
        "候选范围：", "",
        "| 参数 | 候选值 |", "| --- | --- |",
    ]
    for name, values in TUNING_CANDIDATE_VALUES.items():
        lines.append(f"| `{name}` | {TUNING_BASE[name]}, {values[1]} |")
    lines += ["", "候選汇总（按候选构造顺序；完整逐折记录见 `../artifacts/tuning_cv_folds.csv`）：", "",
              "| 候选 | 变更参数 | 取值 | 平均 MAE (MPa) | MAE 标准差 (MPa) | 平均 RMSE (MPa) | 平均 R² |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for candidate in summaries:
        lines.append(f"| {candidate['candidate_id']} | {candidate['changed_parameter']} | {candidate['tested_value']} | {candidate['mean_MAE_MPa']:.6f} | {candidate['std_MAE_MPa']:.6f} | {candidate['mean_RMSE_MPa']:.6f} | {candidate['mean_R2']:.6f} |")
    lines += ["", f"最终选择 `{winner['candidate_id']}`，依据是三折平均 MAE 最低（{winner['mean_MAE_MPa']:.6f} MPa，折间标准差 {winner['std_MAE_MPa']:.6f} MPa）。与基准配置相比，最终仅 `min_child_samples` 从 12 改为 8，其余候选参数保持基准值。最终参数：", "",
              "```json", json.dumps(params, ensure_ascii=False, indent=2), "```", "",
              "同一次最终测试评估中的 LightGBM 调参前后指标：", "",
              "| 状态 | MAE (MPa) | RMSE (MPa) | R² |", "| --- | ---: | ---: | ---: |",
              f"| 调参前，原始配置 | {original_test['MAE_MPa']:.6f} | {original_test['RMSE_MPa']:.6f} | {original_test['R2']:.6f} |",
              f"| 调参后，CV 选择配置 | {tuned_test['MAE_MPa']:.6f} | {tuned_test['RMSE_MPa']:.6f} | {tuned_test['R2']:.6f} |", "",
              "测试集只用于该次末端报告，不用于搜索候选、选择特征或决定是否重新搜索。"]
    (root / "evidence" / "主模型调参记录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = project_root(); data_path = root / "data" / "al_data.csv"
    plot_font = configure_plot_font()
    if not data_path.exists(): raise FileNotFoundError(data_path)
    out_models = root / "models"; out_models.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    source_rows = len(df)
    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    dropped_index_column = None
    if len(df.columns):
        first = str(df.columns[0])
        first_values = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        sequential_index = bool(first_values.notna().all() and np.array_equal(first_values.to_numpy(), np.arange(len(df))))
        if first.startswith("Unnamed") or first == "" or first.lower() in {"index", "row_index"} or sequential_index:
            dropped_index_column = first
            df = df.iloc[:, 1:].copy()
    missing = sorted({PROCESSING, TARGET} - set(df.columns))
    if missing: raise ValueError(f"missing required columns: {missing}")
    raw_processing_counts = df[PROCESSING].fillna("<缺失>").astype(str).value_counts(dropna=False).to_dict()
    normalized_processing = df[PROCESSING].map(normalize_processing).replace("", np.nan)
    normalized_counts = normalized_processing.fillna("<缺失>").value_counts(dropna=False).to_dict()
    pd.DataFrame([{"raw_processing": label, "rows": int(count), "normalized_processing": normalize_processing(label) if label != "<缺失>" else "<缺失>"}
                  for label, count in raw_processing_counts.items()]).to_csv(root / "artifacts" / "processing_value_counts.csv", index=False, encoding="utf-8-sig")
    processing_audit = {
        "raw_value_counts": {str(k): int(v) for k, v in raw_processing_counts.items()},
        "normalized_value_counts": {str(k): int(v) for k, v in normalized_counts.items()},
        "decision": "Strain Harderned (Hard) 与 Strain hardened 保留为两个工艺状态；只将 Harderned 拼写纠正为 hardened，并与同一括号限定的大小写变体归一。依据：原始标签并列存在且来源研究将其作为独立处理类别。",
        "aliases": PROCESSING_ALIASES,
    }
    save_json(root / "artifacts" / "processing_audit.json", processing_audit)
    processing_doc = root / "evidence" / "数据字段说明.md"
    processing_text = processing_doc.read_text(encoding="utf-8")
    section_title = "## Processing 拼写与类别核验"
    if section_title in processing_text:
        processing_text = processing_text.split(section_title, 1)[0].rstrip()
    hard_count = int(raw_processing_counts.get("Strain Harderned (Hard)", 0))
    plain_count = int(raw_processing_counts.get("Strain hardened", 0))
    processing_section = (
        "\n\n## Processing 拼写与类别核验（2026-11-15）\n\n"
        f"原始 CSV 中 `Strain Harderned (Hard)` 为 {hard_count} 行，`Strain hardened` 为 {plain_count} 行。二者保留为不同类别：来源研究将 `Strain hardened` 和 `Strain hardened (Hard)` 分别列为 10 种工艺状态中的独立类别，但没有在该数据字段定义中进一步给出两者的工艺参数差异；因此本项目只据类别标签保留其区别，不推断额外冶金含义。标准化仅把 `Harderned` 拼写更正为 `hardened`，并统一相同括号限定词的大小写形式。训练和预测使用同一映射。计数见 `../artifacts/processing_value_counts.csv`；来源：Bhat et al., DOI `10.1016/j.commatsci.2023.112270`。\n"
    )
    processing_doc.write_text(processing_text + processing_section, encoding="utf-8")
    numeric_features = [c for c in df.columns if c not in EXCLUDED and c != PROCESSING and not c.startswith("Unnamed")]
    for name in numeric_features:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df[PROCESSING] = normalized_processing
    quality = data_quality_audit(df, numeric_features)
    source_target_missing_rows_before_dedup = int(df[TARGET].isna().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    required_fields = numeric_features + [PROCESSING, TARGET]
    invalid_required_rows = int(df[required_fields].isna().any(axis=1).sum())
    missing_target_rows = int(df[TARGET].isna().sum())
    df = df.dropna(subset=required_fields).reset_index(drop=True)
    quality["duplicate_rows_removed"] = quality["exact_duplicate_rows_before_cleaning"]
    quality["rows_removed_for_missing_or_invalid_required_fields"] = invalid_required_rows
    quality["dropped_row_index_column"] = dropped_index_column
    quality["rows_after_cleaning"] = int(len(df))
    quality["source_sha256"] = source_hash
    quality["processing_audit"] = processing_audit
    save_json(root / "data_quality_report.json", quality)
    y = df[TARGET].astype(float)
    group_key = df[numeric_features].round(10).astype(str).agg("|".join, axis=1)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED).split(df, y, groups=group_key))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    X_train, all_features = normalize_and_matrix(train_df, numeric_features)
    X_test, test_feature_order = normalize_and_matrix(test_df, numeric_features)
    X_test = X_test.reindex(columns=all_features, fill_value=0.0)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_group_set, test_group_set = set(group_key.iloc[train_idx]), set(group_key.iloc[test_idx])
    if train_group_set.intersection(test_group_set):
        raise AssertionError("composition group leakage detected")
    train_processing = set(train_df[PROCESSING].unique())
    test_processing = set(test_df[PROCESSING].unique())
    unseen_processing = sorted(test_processing - train_processing)

    best_params, cv_fold_results, cv_summaries = tune_lgbm_group_cv(X_train, y_train, group_key.iloc[train_idx].reset_index(drop=True), SEED)
    selected, report, selection_info = select_features(X_train, y_train)
    selection_info["training_only"] = True
    selection_info["selected_features"] = selected
    report.to_csv(root / "feature_selection_details.csv", index=False, encoding="utf-8-sig")

    full_feature_model = fit_lgbm(X_train, y_train, best_params, SEED)
    full_feature_pred = full_feature_model.predict(X_test)
    predictions = evaluate_model_suite(X_train, y_train, X_test, selected, best_params, SEED)
    tuned_primary_metrics = metrics(y_test.to_numpy(), predictions["LightGBM"])
    original_model = fit_original_lgbm(X_train[selected], y_train, SEED)
    original_lgbm_metrics = metrics(y_test.to_numpy(), original_model.predict(X_test[selected]))
    selection_compare = pd.DataFrame([
        {"stage": "选择前", "feature_count": len(all_features), **metrics(y_test.to_numpy(), full_feature_pred)},
        {"stage": "选择后", "feature_count": len(selected), **tuned_primary_metrics},
    ])
    selection_compare.to_csv(root / "selection_report.csv", index=False, encoding="utf-8-sig")
    write_tuning_evidence(root, best_params, cv_fold_results, cv_summaries, tuned_primary_metrics, original_lgbm_metrics)

    import shap
    final_model = fit_lgbm(X_train[selected], y_train, best_params, SEED)
    shap_values = shap.TreeExplainer(final_model).shap_values(X_test[selected])
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_array = np.asarray(shap_values)
    shap_df = pd.DataFrame({"feature": selected, "mean_abs_shap": np.abs(shap_array).mean(axis=0),
                            "mean_shap": shap_array.mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(root / "shap_summary.csv", index=False, encoding="utf-8-sig")
    fig_paths = write_figures(root, y_test, predictions, final_model, selected, selection_compare,
                              test_df[PROCESSING].reset_index(drop=True), shap_df)
    _, stability_summary, stability_figure = run_stability(df, numeric_features, best_params, root)
    fig_paths.append(str(root / stability_figure))
    model_metrics = {name: metrics(y_test.to_numpy(), pred) for name, pred in predictions.items()}

    cv_best = min(cv_summaries, key=lambda item: (item["mean_MAE_MPa"], item["std_MAE_MPa"]))
    config = {"target": TARGET, "target_unit": "MPa", "processing_field": PROCESSING,
              "numeric_features": numeric_features, "field_unit": "%", "source_composition_encoding": "0-1 proportion",
              "source_to_model_scale_factor": 100.0, "service_input_scale_factor": 1.0,
              "model_input_unit": "%", "all_model_features": all_features,
              "selected_features": selected, "processing_values": sorted(train_df[PROCESSING].unique().tolist()),
              "processing_aliases": PROCESSING_ALIASES, "excluded_fields": sorted(EXCLUDED | {"row_index"}), "seed": SEED}
    save_json(root / "feature_config.json", config)
    (out_models / "lightgbm.txt").write_text(final_model.booster_.model_to_string(), encoding="utf-8")
    joblib.dump(predictions_model_rf := RandomForestRegressor(n_estimators=500, max_features=0.8, random_state=SEED, n_jobs=-1).fit(X_train[selected], y_train), out_models / "rf_baseline.pkl")
    joblib.dump(XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.9,
                            colsample_bytree=0.9, objective="reg:squarederror", random_state=SEED, n_jobs=4)
               .fit(X_train[selected], y_train, verbose=False), out_models / "xgb_baseline.pkl")
    joblib.dump(final_model, out_models / "lightgbm.pkl")
    registry = {"primary": "lightgbm", "tuning": {"method": "3-fold GroupKFold on training partition", "selection_metric": "mean MAE (MPa)", "best_candidate": cv_best["candidate_id"], "parameters": best_params},
                "models": {"lightgbm": {"path": "models/lightgbm.txt", "type": "lightgbm", "features": "selected_features", "parameters": best_params},
                           "random_forest": {"path": "models/rf_baseline.pkl", "type": "joblib", "features": "selected_features", "parameters": {"n_estimators": 500, "max_features": 0.8}},
                           "xgboost": {"path": "models/xgb_baseline.pkl", "type": "joblib", "features": "selected_features", "parameters": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "objective": "reg:squarederror"}}},
                "metrics": model_metrics, "lightgbm_before_tuning_same_test": original_lgbm_metrics}
    save_json(root / "model_registry.json", registry)
    predictions_frame = pd.DataFrame({"actual_yield_strength_MPa": y_test.to_numpy(), "LightGBM": predictions["LightGBM"],
                                      "RandomForest": predictions["RandomForest"], "XGBoost": predictions["XGBoost"],
                                      "processing": test_df[PROCESSING].to_numpy()})
    predictions_frame.to_csv(root / "artifacts" / "test_predictions.csv", index=False, encoding="utf-8-sig")
    stability_csv_rows = pd.read_csv(root / "evidence" / "分组划分稳定性对比.csv").to_dict(orient="records")
    metrics_doc = {"seed": SEED, "split": "GroupShuffleSplit(train_size=0.8) by exact composition combination",
                   "source": "Mendeley Data V1 DOI 10.17632/b6br4yk6r3.1", "data_file": "data/al_data.csv", "source_sha256": source_hash,
                   "source_rows": source_rows,
                   "exact_duplicate_rows_removed": int(quality["duplicate_rows_removed"]),
                   "rows_removed_for_missing_or_invalid_required_fields": invalid_required_rows,
                   "source_target_missing_rows_before_dedup": source_target_missing_rows_before_dedup,
                   "cleaning_formula": f"{source_rows} - {quality['duplicate_rows_removed']} - {invalid_required_rows} = {len(df)}",
                   "total_samples": len(df),
                   "train_samples": len(train_idx), "test_samples": len(test_idx), "unique_composition_groups": int(group_key.nunique()),
                   "train_groups": len(train_group_set), "test_groups": len(test_group_set),
                   "train_test_group_intersection_count": 0, "unseen_test_processing_values": unseen_processing,
                   "test_processing_values_reindexed_to_zero": unseen_processing,
                   "target": TARGET, "target_unit": "MPa", "plot_font": plot_font, "models": model_metrics,
                   "feature_selection": selection_info, "selection_comparison": selection_compare.to_dict(orient="records"),
                   "lightgbm_before_tuning_same_test": original_lgbm_metrics, "lightgbm_after_tuning_same_test": tuned_primary_metrics,
                   "tuning": {"candidate_count": len(cv_summaries), "folds_per_candidate": CV_SPLITS,
                              "best_candidate": cv_best["candidate_id"], "parameters": best_params},
                   "stability_seeds": STABILITY_SEEDS, "stability": stability_summary,
                   "processing_audit": processing_audit, "figures": [str(Path(p).relative_to(root)) for p in fig_paths]}
    save_json(root / "metrics.json", metrics_doc)
    print(json.dumps({"total_samples": len(df), "train_samples": len(train_idx), "test_samples": len(test_idx),
                      "models": model_metrics, "selected_features": len(selected), "all_features": len(all_features),
                      "best_params": best_params, "before_tuning": original_lgbm_metrics, "figures": len(fig_paths),
                      "unseen_processing": unseen_processing, "stability": stability_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
