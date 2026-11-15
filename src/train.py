from __future__ import annotations

import json
import hashlib
import re
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
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

SEED = 2024
TARGET = "Yield Strength (MPa)"
PROCESSING = "Processing"
EXCLUDED = {"Elongation (%)", "Tensile Strength (MPa)", "class", TARGET}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_processing(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"\s*\+\s*", " + ", text)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE_MPa": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression", n_estimators=700, learning_rate=0.035,
        num_leaves=31, min_child_samples=12, subsample=0.9,
        colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=0.2,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def configure_plot_font() -> str:
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


def fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    model = make_lgbm()
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


def select_features(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    variances = X_train.var(axis=0)
    variance_keep = [c for c in X_train.columns if float(variances.get(c, 0.0)) > 1e-12]
    corr = X_train[variance_keep].corr().abs()
    corr_keep: list[str] = []
    for col in variance_keep:
        if not corr_keep or not any(float(corr.loc[col, prior]) >= 0.98 for prior in corr_keep):
            corr_keep.append(col)
    pre_model = make_lgbm()
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
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(actual, pred, s=28, alpha=.75, color="#D58A00"); lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max())); ax.plot([lo, hi], [lo, hi], "--", color="#2C2C2A"); ax.set(title="真实值与预测值", xlabel="真实屈服强度 (MPa)", ylabel="预测屈服强度 (MPa)"); fig.tight_layout(); p=fig_dir/"01_actual_vs_pred.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    errors = pred - actual
    fig, ax = plt.subplots(figsize=(8, 6)); sns.histplot(errors, kde=True, ax=ax, color="#E8A317"); ax.axvline(0, ls="--", color="#2C2C2A"); ax.set(title="预测误差分布", xlabel="预测误差 (MPa)", ylabel="样本数"); fig.tight_layout(); p=fig_dir/"02_error_distribution.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    imp = pd.Series(final_model.feature_importances_, index=selected).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(9, 7)); imp.plot.barh(ax=ax, color="#E8A317"); ax.set(title="LightGBM 特征重要性", xlabel="分裂重要性（无量纲相对值）", ylabel="输入特征"); fig.tight_layout(); p=fig_dir/"03_feature_importance.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for metric_name, ax in zip(["MAE_MPa", "RMSE_MPa", "R2"], axes):
        label = metric_name.replace("_MPa", " (MPa)") if metric_name != "R2" else "R²（无量纲）"
        sns.barplot(data=selection_compare, x="stage", y=metric_name, ax=ax, color="#E8A317"); ax.set(title=label, xlabel="特征阶段", ylabel=label); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("特征选择前后评价对比"); fig.tight_layout(); p=fig_dir/"04_feature_selection_comparison.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    top = shap_df.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top["feature"], top["mean_abs_shap"], color="#D58A00"); ax.set(title="SHAP 全局贡献", xlabel="平均绝对 SHAP 值 (MPa)", ylabel="输入特征"); fig.tight_layout(); p=fig_dir/"05_shap_global.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    grouped = pd.DataFrame({"Processing": group_labels.to_numpy(), "error": errors}).groupby("Processing", as_index=False).agg(mean_abs_error=("error", lambda x: float(np.mean(np.abs(x)))), count=("error", "size")).sort_values("mean_abs_error", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(grouped["Processing"].astype(str)[::-1], grouped["mean_abs_error"][::-1], color="#E8A317"); ax.set(title="按 Processing 分组的误差", xlabel="平均绝对误差 (MPa)", ylabel="Processing（处理状态）"); fig.tight_layout(); p=fig_dir/"06_grouped_error.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    md = pd.DataFrame([{"model": name, **metrics(actual, vals)} for name, vals in predictions.items()])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for metric_name, ax in zip(["MAE_MPa", "RMSE_MPa", "R2"], axes):
        label = metric_name.replace("_MPa", " (MPa)") if metric_name != "R2" else "R²（无量纲）"
        ax.bar(md["model"], md[metric_name], color=["#E8A317", "#F2C565", "#D58A00"]); ax.set(title=label, xlabel="模型", ylabel=label); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("三模型测试集指标对比"); fig.tight_layout(); p=fig_dir/"07_model_comparison.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    return paths


def main() -> None:
    root = project_root(); data_path = root / "data" / "al_data.csv"
    plot_font = configure_plot_font()
    if not data_path.exists(): raise FileNotFoundError(data_path)
    out_models = root / "models"; out_models.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    source_rows = len(df)
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
    numeric_features = [c for c in df.columns if c not in EXCLUDED and c != PROCESSING and not c.startswith("Unnamed")]
    for name in numeric_features:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df[PROCESSING] = df[PROCESSING].map(normalize_processing).replace("", np.nan)
    quality = data_quality_audit(df, numeric_features)
    df = df.drop_duplicates().reset_index(drop=True)
    required_fields = numeric_features + [PROCESSING, TARGET]
    invalid_required_rows = int(df[required_fields].isna().any(axis=1).sum())
    missing_target_rows = int(df[TARGET].isna().sum())
    df = df.dropna(subset=required_fields).reset_index(drop=True)
    quality["duplicate_rows_removed"] = quality["exact_duplicate_rows_before_cleaning"]
    quality["rows_removed_for_missing_or_invalid_required_fields"] = invalid_required_rows
    quality["dropped_row_index_column"] = dropped_index_column
    quality["rows_after_cleaning"] = int(len(df))
    save_json(root / "data_quality_report.json", quality)
    y = df[TARGET].astype(float)
    group_key = df[numeric_features].round(10).astype(str).agg("|".join, axis=1)
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED); train_idx, test_idx = next(splitter.split(df, y, groups=group_key))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    X_train, all_features = normalize_and_matrix(train_df, numeric_features)
    X_test, _ = normalize_and_matrix(test_df, numeric_features)
    X_test = X_test.reindex(columns=all_features, fill_value=0.0)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if set(group_key.iloc[train_idx]).intersection(set(group_key.iloc[test_idx])):
        raise AssertionError("composition group leakage detected")
    selected, report, selection_info = select_features(X_train, y_train); report.to_csv(root / "feature_selection_details.csv", index=False, encoding="utf-8-sig")
    before_model = fit_lgbm(X_train, y_train); before_pred = before_model.predict(X_test)
    final_model = fit_lgbm(X_train[selected], y_train); lgbm_pred = final_model.predict(X_test[selected])
    rf = RandomForestRegressor(n_estimators=500, max_features=0.8, random_state=SEED, n_jobs=-1); rf.fit(X_train[selected], y_train); rf_pred = rf.predict(X_test[selected])
    xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=SEED, n_jobs=4); xgb.fit(X_train[selected], y_train, verbose=False); xgb_pred = xgb.predict(X_test[selected])
    predictions = {"LightGBM": lgbm_pred, "RandomForest": rf_pred, "XGBoost": xgb_pred}
    import shap
    shap_values = shap.TreeExplainer(final_model).shap_values(X_test[selected]); shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_df = pd.DataFrame({"feature": selected, "mean_abs_shap": np.abs(np.asarray(shap_values)).mean(axis=0), "mean_shap": np.asarray(shap_values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False); shap_df.to_csv(root / "shap_summary.csv", index=False, encoding="utf-8-sig")
    selection_compare = pd.DataFrame([{"stage":"选择前", "feature_count":len(all_features), **metrics(y_test, before_pred)}, {"stage":"选择后", "feature_count":len(selected), **metrics(y_test, lgbm_pred)}])
    selection_compare.to_csv(root / "selection_report.csv", index=False, encoding="utf-8-sig")
    fig_paths = write_figures(root, y_test, predictions, final_model, selected, selection_compare, df.iloc[test_idx][PROCESSING].reset_index(drop=True), shap_df)
    model_metrics = {name: metrics(y_test, pred) for name, pred in predictions.items()}
    save_json(root / "metrics.json", {"seed":SEED, "source":"Mendeley Data V1 DOI 10.17632/b6br4yk6r3.1", "data_file":"data/al_data.csv", "source_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(), "source_rows":source_rows, "dropped_missing_target_rows":missing_target_rows, "total_samples":len(df), "train_samples":len(train_idx), "test_samples":len(test_idx), "unique_composition_groups":int(group_key.nunique()), "train_groups":int(group_key.iloc[train_idx].nunique()), "test_groups":int(group_key.iloc[test_idx].nunique()), "target":TARGET, "target_unit":"MPa", "plot_font":plot_font, "models":model_metrics, "feature_selection":selection_info, "selection_comparison":selection_compare.to_dict(orient="records"), "figures":[str(Path(p).relative_to(root)) for p in fig_paths]})
    save_json(root / "feature_config.json", {"target":TARGET, "target_unit":"MPa", "processing_field":PROCESSING, "numeric_features":numeric_features, "field_unit":"%", "source_composition_encoding":"0-1 proportion", "source_to_model_scale_factor":100.0, "model_input_unit":"%", "all_model_features":all_features, "selected_features":selected, "processing_values":sorted(train_df[PROCESSING].unique().tolist()), "excluded_fields":sorted(EXCLUDED | {"row_index"}), "seed":SEED})
    (out_models / "lightgbm.txt").write_text(final_model.booster_.model_to_string(), encoding="utf-8")
    joblib.dump(rf, out_models / "rf_baseline.pkl"); joblib.dump(xgb, out_models / "xgb_baseline.pkl"); joblib.dump(final_model, out_models / "lightgbm.pkl")
    save_json(root / "model_registry.json", {"primary":"lightgbm", "models":{"lightgbm":{"path":"models/lightgbm.txt","type":"lightgbm","features":"selected_features"}, "random_forest":{"path":"models/rf_baseline.pkl","type":"joblib","features":"selected_features"}, "xgboost":{"path":"models/xgb_baseline.pkl","type":"joblib","features":"selected_features"}}, "metrics":model_metrics})
    pd.DataFrame({"actual_yield_strength_MPa":y_test.to_numpy(), "LightGBM":lgbm_pred, "RandomForest":rf_pred, "XGBoost":xgb_pred, "processing":df.iloc[test_idx][PROCESSING].to_numpy()}).to_csv(root / "artifacts" / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"total_samples":len(df), "train_samples":len(train_idx), "test_samples":len(test_idx), "models":model_metrics, "selected_features":len(selected), "figures":len(fig_paths)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
