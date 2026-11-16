from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.predict_service import (  # noqa: E402
    clear_records,
    explain_record,
    list_records,
    load_config,
    load_model,
    predict_dataframe,
    predict_record,
)
from service.comparison_service import comparison_csv, comparison_frame, comparison_markdown  # noqa: E402
from service.sensitivity_service import build_compensated_scan  # noqa: E402
from theme import (  # noqa: E402
    BORDER,
    MODEL_COLORS,
    PLOT_DPI,
    PLOT_GRID,
    PRIMARY,
    PURPLE,
    SECONDARY_BACKGROUND,
    TEAL,
    TEXT,
    WARNING,
    configure_plot_font,
)


st.set_page_config(page_title="铝合金屈服强度预测", layout="wide", initial_sidebar_state="expanded")
PLOT_FONT = configure_plot_font()
CFG = load_config()
METRICS = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
REGISTRY = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))
RANGES = json.loads((ROOT / "artifacts" / "sensitivity_ranges.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="正在加载已训练模型…")
def _cached_model(model_name: str = "lightgbm"):
    return load_model(model_name)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def _cached_csv(path: Path) -> pd.DataFrame:
    return _read_csv(str(path), path.stat().st_mtime_ns)


def header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


@st.cache_data(show_spinner=False)
def _clean_source() -> pd.DataFrame:
    data = _cached_csv(ROOT / "data" / "al_data.csv")
    if len(data.columns):
        first = str(data.columns[0])
        values = pd.to_numeric(data.iloc[:, 0], errors="coerce")
        sequential = bool(values.notna().all() and np.array_equal(values.to_numpy(), np.arange(len(data))))
        if first.startswith("Unnamed") or first == "" or first.lower() in {"index", "row_index"} or sequential:
            data = data.iloc[:, 1:].copy()
    for name in CFG["numeric_features"] + [CFG["target"]]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    aliases = CFG.get("processing_aliases", {})
    data[CFG["processing_field"]] = data[CFG["processing_field"]].map(
        lambda value: aliases.get(str(value).strip().casefold(), str(value).strip())
    )
    required = CFG["numeric_features"] + [CFG["processing_field"], CFG["target"]]
    data = data.drop_duplicates().dropna(subset=required).reset_index(drop=True)
    data[CFG["numeric_features"]] *= float(CFG["source_to_model_scale_factor"])
    return data


def _plot_style(ax: plt.Axes) -> None:
    ax.grid(True, color=PLOT_GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=TEXT)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def data_management() -> None:
    header("数据管理", "查看唯一数据源、字段单位和清洗后的输入口径")
    data = _cached_csv(ROOT / "data" / "al_data.csv")
    if len(data.columns) and (str(data.columns[0]).startswith("Unnamed") or str(data.columns[0]) == ""):
        data = data.iloc[:, 1:].copy()
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("源数据行数", f"{len(data):,}")
        c2.metric("源字段数", len(data.columns))
        c3.metric("目标单位", METRICS["target_unit"])
    st.divider()
    st.subheader("数据预览")
    display_df = data.copy()
    for name in CFG["numeric_features"]:
        display_df[name] = pd.to_numeric(display_df[name], errors="coerce") * CFG["source_to_model_scale_factor"]
    st.info("源 CSV 的成分以 0–1 比例存储；本页预览、训练及预测输入统一显示为百分比。")
    st.dataframe(display_df.head(30), width="stretch", height=340)
    with st.container(border=True):
        st.subheader("字段与处理")
        st.write("保留成分字段与 Processing；移除行索引、Elongation、Tensile Strength、class。无标签新样本只做预测，不计算误差。")
        st.dataframe(
            pd.DataFrame(
                {
                    "字段": CFG["numeric_features"] + [CFG["processing_field"]],
                    "单位": ["%（源 CSV 比例值×100）"] * len(CFG["numeric_features"]) + ["处理状态枚举"],
                }
            ),
            width="stretch",
            hide_index=True,
        )


def _model_metric_cards() -> None:
    rf = METRICS["models"]["RandomForest"]
    with st.container(border=True):
        columns = st.columns(3)
        for column, (name, values) in zip(columns, METRICS["models"].items()):
            delta = values["MAE_MPa"] - rf["MAE_MPa"] if name != "RandomForest" else 0.0
            delta_text = f"{delta:+.2f} vs RF" if name != "RandomForest" else "基线"
            column.metric(f"{name} MAE", f"{values['MAE_MPa']:.2f} MPa", delta=delta_text, delta_color="inverse")
            column.caption(f"RMSE {values['RMSE_MPa']:.2f} MPa · R² {values['R2']:.4f}")


def _show_image(name: str) -> None:
    if name == "05_shap_global.png":
        st.image(str(ROOT / "artifacts" / "figures" / name), width="stretch")
        return
    predictions = _cached_csv(ROOT / "artifacts" / "test_predictions.csv")
    figure, ax = plt.subplots(figsize=(9, 5.4), dpi=PLOT_DPI)
    if name == "01_actual_vs_pred.png":
        actual = predictions["actual_yield_strength_MPa"]
        predicted = predictions["LightGBM"]
        ax.scatter(actual, predicted, color=MODEL_COLORS["LightGBM"], alpha=0.72, s=24)
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        ax.plot([low, high], [low, high], linestyle="--", color=TEXT, label="y = x")
        primary = METRICS["models"]["LightGBM"]
        ax.text(0.04, 0.96, f"MAE = {primary['MAE_MPa']:.2f} MPa\nR² = {primary['R2']:.4f}", transform=ax.transAxes, va="top", color=TEXT)
        ax.set_title("真实值与预测值")
        ax.set_xlabel("真实屈服强度 (MPa)")
        ax.set_ylabel("预测屈服强度 (MPa)")
        ax.legend()
    elif name == "02_error_distribution.png":
        errors = predictions["LightGBM"] - predictions["actual_yield_strength_MPa"]
        ax.hist(errors, bins=24, color=MODEL_COLORS["LightGBM"], edgecolor=TEXT, alpha=0.85)
        ax.axvline(0, linestyle="--", color=TEXT)
        ax.set_title("预测误差分布")
        ax.set_xlabel("预测误差 (MPa)")
        ax.set_ylabel("样本数")
    elif name == "03_feature_importance.png":
        details = _cached_csv(ROOT / "feature_selection_details.csv").sort_values("gain").tail(20)
        ax.barh(details["feature"], details["gain"], color=MODEL_COLORS["LightGBM"])
        ax.set_title("LightGBM 特征重要性")
        ax.set_xlabel("训练集 gain（相对值）")
        ax.set_ylabel("输入特征")
    elif name == "04_feature_selection_comparison.png":
        comparison = pd.DataFrame(METRICS["selection_comparison"])
        x = np.arange(len(comparison))
        width = 0.24
        series = [("MAE_MPa", "MAE (MPa)", PRIMARY), ("RMSE_MPa", "RMSE (MPa)", PURPLE), ("R2", "R²", TEAL)]
        for index, (column, label, color) in enumerate(series):
            ax.bar(x + (index - 1) * width, comparison[column], width, label=label, color=color)
        ax.set_xticks(x, comparison["stage"])
        ax.set_title("特征选择前后评价对比")
        ax.set_xlabel("特征阶段")
        ax.set_ylabel("指标值（误差 MPa；R² 无量纲）")
        ax.legend()
    elif name == "06_grouped_error.png":
        grouped = predictions.assign(error=(predictions["LightGBM"] - predictions["actual_yield_strength_MPa"]).abs())
        grouped = grouped.groupby("processing", as_index=False).agg(error=("error", "mean"), count=("error", "size"))
        grouped = grouped.sort_values("error").tail(15)
        cmap = LinearSegmentedColormap.from_list("semantic_yellow_purple", [PRIMARY, PURPLE])
        colors = [cmap(value) for value in np.linspace(0, 1, len(grouped))]
        ax.barh(grouped["processing"], grouped["error"], color=colors)
        ax.set_title("按 Processing 分组的误差")
        ax.set_xlabel("平均绝对误差 (MPa)")
        ax.set_ylabel("Processing（处理状态）")
    elif name == "07_model_comparison.png":
        model_metrics = METRICS["models"]
        model_names = list(model_metrics)
        x = np.arange(len(model_names))
        width = 0.24
        for index, (column, label) in enumerate((("MAE_MPa", "MAE (MPa)"), ("RMSE_MPa", "RMSE (MPa)"), ("R2", "R²"))):
            ax.bar(x + (index - 1) * width, [model_metrics[name][column] for name in model_names], width,
                   label=label, color=[MODEL_COLORS[name] for name in model_names], alpha=(1.0 - index * 0.12))
        ax.set_xticks(x, model_names)
        ax.set_title("三模型测试集指标对比")
        ax.set_xlabel("模型")
        ax.set_ylabel("指标值（误差 MPa；R² 无量纲）")
        ax.legend()
    elif name == "08_stability.png":
        stability = METRICS["stability"]
        models = list(stability)
        x = np.arange(len(models))
        metrics = (("MAE_MPa", "MAE (MPa)"), ("RMSE_MPa", "RMSE (MPa)"), ("R2", "R²"))
        for metric_index, (metric_name, metric_label) in enumerate(metrics):
            values = [stability[model][metric_name]["mean"] for model in models]
            deviations = [stability[model][metric_name]["std"] for model in models]
            ax.errorbar(x + (metric_index - 1) * 0.12, values, yerr=deviations, fmt="o", capsize=4,
                        color=[PRIMARY, PURPLE, TEAL][metric_index], label=metric_label)
        ax.set_xticks(x, models)
        ax.set_title("分组划分稳定性（均值 ± 标准差）")
        ax.set_xlabel("模型")
        ax.set_ylabel("指标值（误差 MPa；R² 无量纲）")
        ax.legend()
    elif name == "09_threshold_sensitivity.png":
        threshold = _cached_csv(ROOT / "evidence" / "特征选择阈值敏感性.csv")
        ax.plot(threshold["gain_retention_ratio"] * 100, threshold["MAE_MPa"], marker="o", color=PRIMARY, label="MAE (MPa)")
        ax.plot(threshold["gain_retention_ratio"] * 100, threshold["RMSE_MPa"], marker="s", color=PURPLE, label="RMSE (MPa)")
        ax.set_title("Gain 保留比例敏感性")
        ax.set_xlabel("Gain 保留比例 (%)")
        ax.set_ylabel("误差 (MPa)")
        ax.legend()
    else:
        plt.close(figure)
        st.error(f"未配置图表：{name}")
        return
    _plot_style(ax)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def model_evaluation() -> None:
    header("模型与评价", "读取已定稿指标，分组展示预测效果、特征解释和附加实验")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("训练样本", METRICS["train_samples"])
        c2.metric("测试样本", METRICS["test_samples"])
        c3.metric("最终特征数", METRICS["feature_selection"]["selected_feature_count"])
    _model_metric_cards()
    st.divider()
    tabs = st.tabs(["预测效果", "特征与解释", "稳定性与阈值"])
    with tabs[0]:
        st.dataframe(pd.DataFrame(METRICS["models"]).T, width="stretch")
        _show_image("01_actual_vs_pred.png")
        _show_image("02_error_distribution.png")
        _show_image("07_model_comparison.png")
    with tabs[1]:
        st.dataframe(pd.DataFrame(METRICS["selection_comparison"]), width="stretch", hide_index=True)
        st.caption(f"特征选择：{METRICS['feature_selection']['initial_feature_count']} → {METRICS['feature_selection']['selected_feature_count']} 项")
        _show_image("03_feature_importance.png")
        _show_image("04_feature_selection_comparison.png")
        _show_image("05_shap_global.png")
        _show_image("06_grouped_error.png")
    with tabs[2]:
        _show_image("08_stability.png")
        _show_image("09_threshold_sensitivity.png")
        st.dataframe(_cached_csv(ROOT / "evidence" / "特征选择阈值敏感性.csv"), width="stretch", hide_index=True)


def _fill_example(prefix: str) -> None:
    example = _cached_csv(ROOT / "outputs" / "sample_input.csv").iloc[0]
    st.session_state[prefix + "_processing"] = str(example[CFG["processing_field"]])
    for name in CFG["numeric_features"]:
        st.session_state[prefix + "_" + name] = float(example[name])


def _numeric_form(prefix: str, with_example: bool = False) -> tuple[str, dict[str, float], bool]:
    if with_example:
        st.button("填入典型合金示例", key=prefix + "_fill_example", on_click=_fill_example, args=(prefix,))
    with st.form(prefix):
        processing_key = prefix + "_processing"
        options = {"label": "Processing 状态", "options": CFG["processing_values"], "key": processing_key}
        if processing_key not in st.session_state:
            options["index"] = 0
        processing = st.selectbox(**options)
        columns = st.columns(4)
        values: dict[str, float] = {}
        for index, name in enumerate(CFG["numeric_features"]):
            key = prefix + "_" + name
            input_options = {"label": name + " (%)", "min_value": 0.0, "max_value": 100.0, "format": "%.6f", "key": key}
            if key not in st.session_state:
                input_options["value"] = 0.0
            values[name] = columns[index % 4].number_input(**input_options)
        submitted = st.form_submit_button("提交", type="primary")
    return processing, values, submitted


def _render_sensitivity(values: dict[str, float], processing: str) -> None:
    st.divider()
    st.subheader("单因素敏感性曲线")
    st.caption(f"模型响应曲线，非物理规律，不代表因果关系。扫描通过 Al 补偿，使 {len(CFG['numeric_features'])} 个元素成分合计保持 100%。")
    selectable = [name for name in CFG["numeric_features"] if name != "Al"]
    element = st.selectbox("选择扫描元素（Al 为补偿字段）", selectable, index=selectable.index("Cu") if "Cu" in selectable else 0)
    interval = RANGES["ranges"][element]
    c1, c2 = st.columns(2)
    c1.metric("训练集 5% 分位", f"{interval['q05_percent']:.4f}%")
    c2.metric("训练集 95% 分位", f"{interval['q95_percent']:.4f}%")
    points = st.slider("扫描点数", min_value=30, max_value=50, value=41)
    try:
        scan = build_compensated_scan(values, processing, element, interval["q05_percent"], interval["q95_percent"], points)
        result = predict_dataframe(scan, model=_cached_model())
        totals = scan[CFG["numeric_features"]].sum(axis=1)
        if not np.allclose(totals.to_numpy(dtype=float), 100.0, atol=1e-9):
            st.error("补偿扫描成分合计校验失败")
            return
        warnings = result.attrs.get("input_warnings", [])
        if warnings:
            st.error("补偿扫描触发了服务层警告：" + "；".join(warnings))
            return
        figure, ax = plt.subplots(figsize=(8, 4.8), dpi=PLOT_DPI)
        ax.plot(scan[element], result["predicted_yield_strength_MPa"], color=MODEL_COLORS["LightGBM"], linewidth=2.2, label="LightGBM")
        current = float(values[element])
        low, high = float(interval["q05_percent"]), float(interval["q95_percent"])
        left, right = min(low, current), max(high, current)
        # Several trace elements are zero throughout the cleaned training
        # split, so their 5th and 95th percentiles can be identical.  Keep
        # the model scan unchanged but widen the display-only x-axis to avoid
        # Matplotlib's singular-transform warning and preserve a readable
        # chart.
        if np.isclose(left, right):
            pad = max(abs(left) * 0.05, 0.5)
            left, right = left - pad, right + pad
        ax.set_xlim(left, right)
        if current < low or current > high:
            ax.axvspan(min(current, low), max(current, low), color=SECONDARY_BACKGROUND, alpha=0.7, label="外推区，仅供参考")
            ax.axvline(current, color=WARNING, linestyle="--", linewidth=1.3)
        ax.set_title(f"模型预测屈服强度随 {element} 变化（其余成分由 Al 补偿）")
        ax.set_xlabel(f"{element} 成分 (%)")
        ax.set_ylabel("预测屈服强度 (MPa)")
        ax.legend()
        _plot_style(ax)
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.success(f"补偿扫描通过：{len(scan)} 个点的元素成分合计均为 100.000%。")
        if current < low or current > high:
            st.warning("当前输入位于训练集 5%–95% 区间外，图中浅色区域为外推区，仅供参考。")
    except ValueError as exc:
        st.error(str(exc))


def one_prediction() -> None:
    header("单条预测", "输入百分比成分并调用已训练 LightGBM；可加入配方对比并查看响应曲线")
    processing, values, submitted = _numeric_form("single_prediction", with_example=True)
    record = {**values, "Processing": processing}
    if submitted:
        try:
            result = predict_record(record, model=_cached_model())
            st.session_state["single_prediction_last"] = {"record": record, "prediction": result["predicted_yield_strength_MPa"]}
            st.success(f"预测屈服强度：{result['predicted_yield_strength_MPa']:.2f} MPa")
            for warning in result.get("warnings", []):
                st.warning(warning)
        except ValueError as exc:
            st.error(str(exc))
    latest = st.session_state.get("single_prediction_last")
    if latest:
        if st.button("加入对比", type="secondary"):
            recipes = st.session_state.setdefault("comparison_recipes", [])
            if len(recipes) >= 3:
                st.warning("最多保存 3 组配方")
            else:
                recipes.append(latest)
                st.success(f"已加入对比（{len(recipes)}/3）")
        _render_sensitivity(latest["record"], latest["record"]["Processing"])


def batch_prediction() -> None:
    header("批量预测", "上传含 Processing 与百分比成分字段的 CSV 或 Excel；无目标列时只输出预测")
    upload = st.file_uploader("选择文件", type=["csv", "xlsx"])
    if upload:
        try:
            data = pd.read_excel(upload) if upload.name.lower().endswith(".xlsx") else pd.read_csv(upload)
            result = predict_dataframe(data, model=_cached_model())
            for warning in result.attrs.get("input_warnings", []):
                st.warning(warning)
            st.dataframe(result, width="stretch")
            st.download_button("导出预测结果 CSV", result.to_csv(index=False).encode("utf-8-sig"), "yield_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"文件校验失败：{exc}")


def explanations() -> None:
    header("模型解释", "全局贡献来自真实测试集 SHAP；局部解释显示单条输入的特征贡献")
    _show_image("05_shap_global.png")
    st.dataframe(_cached_csv(ROOT / "shap_summary.csv"), width="stretch", hide_index=True)
    _show_image("06_grouped_error.png")
    st.subheader("单条预测局部解释")
    processing, values, submitted = _numeric_form("local_explanation")
    if submitted:
        try:
            explanation = explain_record({**values, "Processing": processing}, model=_cached_model())
            st.metric("屈服强度预测", f"{explanation['prediction_MPa']:.2f} MPa")
            for warning in explanation.get("warnings", []):
                st.warning(warning)
            local = pd.DataFrame(explanation["contributions"])
            st.bar_chart(local.set_index("feature")["shap_value_MPa"])
            st.dataframe(local, width="stretch", hide_index=True)
        except ValueError as exc:
            st.error(str(exc))


def comparison_page() -> None:
    header("配方对比", "最多保存 3 组已计算配方，比较输入差异与模型预测值")
    recipes = st.session_state.get("comparison_recipes", [])
    if not recipes:
        st.info("暂无配方。请在“单条预测”页提交后点击“加入对比”。")
        return
    table = comparison_frame(recipes, CFG["numeric_features"])
    prediction_column = "预测屈服强度 (MPa)"
    best_index = int(table[prediction_column].idxmax())
    styled = table.style.apply(lambda row: [f"background-color: {SECONDARY_BACKGROUND}" if row.name == best_index else "" for _ in row], axis=1)
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={prediction_column: st.column_config.NumberColumn(prediction_column, format="%.1f")},
    )
    st.success(f"当前最高预测值：{table.loc[best_index, prediction_column]:.6f} MPa（{table.loc[best_index, '配方']}）")
    if len(table) > 1:
        base = table.iloc[0]
        differences = []
        for _, row in table.iterrows():
            diff = {"配方": row["配方"], "预测强度差值 (MPa)": float(row[prediction_column] - base[prediction_column])}
            for name in CFG["numeric_features"]:
                diff[f"{name} 差值 (%)"] = float(row[name] - base[name])
            differences.append(diff)
        st.subheader("相对配方 1 的差异")
        st.dataframe(pd.DataFrame(differences), width="stretch", hide_index=True)
    export = comparison_csv(table)
    report = comparison_markdown(table)
    c1, c2, c3 = st.columns(3)
    c1.download_button("导出对比 CSV", export, "recipe_comparison.csv", "text/csv")
    c2.download_button("下载配方报告 Markdown", report, "recipe_comparison.md", "text/markdown")
    if c3.button("清空对比"):
        st.session_state["comparison_recipes"] = []
        st.rerun()


def records_page() -> None:
    header("预测记录", "本地 SQLite records.db；可筛选、查看详情、导出或清空")
    rows = list_records(limit=500)
    records = pd.DataFrame(rows)
    if records.empty:
        st.info("暂无预测记录")
        return
    options = ["全部"] + sorted(records["model"].unique().tolist())
    selected = st.selectbox("模型筛选", options)
    view = records if selected == "全部" else records[records["model"] == selected]
    st.dataframe(view.drop(columns=["input"], errors="ignore"), width="stretch", hide_index=True)
    st.download_button("导出记录 CSV", view.to_csv(index=False).encode("utf-8-sig"), "prediction_records.csv", "text/csv")
    if st.button("清空全部记录"):
        clear_records()
        st.rerun()
    record_id = st.selectbox("记录详情", view["id"].tolist())
    with st.expander("详情", expanded=True):
        st.json(view.loc[view["id"] == record_id].iloc[0].to_dict())


def about() -> None:
    header("关于与说明", "数据、字段、模型和运行方式均从项目交付文件读取")
    with st.container(border=True):
        st.info("预测结果仅用于适用数据范围内的初步分析，不替代标准力学性能试验；新样本无标签时只输出预测值，不计算误差。")
    st.divider()
    with st.container(border=True):
        st.subheader("字段与单位")
        st.write(f"{len(CFG['numeric_features'])} 个元素成分输入单位为百分比 0–100；Processing 为枚举；目标单位为 MPa。")
        st.write("允许的 Processing 值：" + "、".join(CFG["processing_values"]))
        field_doc = (ROOT / "evidence" / "数据字段说明.md").read_bytes()
        st.download_button("下载字段说明", field_doc, "数据字段说明.md", "text/markdown")
    with st.container(border=True):
        st.subheader("模型与实测结果")
        st.write(f"主模型：{REGISTRY['primary']}；基线模型：" + "、".join(name for name in REGISTRY["models"] if name != REGISTRY["primary"]))
        st.write(f"最终特征数：{METRICS['feature_selection']['selected_feature_count']}；清洗后样本数：{METRICS['total_samples']}；训练：{METRICS['train_samples']}；测试：{METRICS['test_samples']}。")
        st.dataframe(pd.DataFrame(METRICS["models"]).T, width="stretch")
        st.success("交付模型已训练完成，网页启动即可预测。")
    st.subheader("屈服强度分布")
    cleaned = _clean_source()
    figure, ax = plt.subplots(figsize=(8, 4.5), dpi=PLOT_DPI)
    ax.hist(cleaned[CFG["target"]], bins=24, color=MODEL_COLORS["LightGBM"], edgecolor=TEXT, alpha=0.85)
    ax.set_title("清洗后屈服强度分布")
    ax.set_xlabel("屈服强度 (MPa)")
    ax.set_ylabel("样本数")
    _plot_style(ax)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.subheader("数据来源与适用范围")
    st.markdown("数据集：Mendeley Data V1，作者 N. Bhat、A. S. Barnard、N. Birbilis；[DOI 10.17632/b6br4yk6r3.1](https://doi.org/10.17632/b6br4yk6r3.1)，许可 CC BY 4.0。")
    st.write("字段依据见配套文章 DOI 10.1016/j.commatsci.2023.112270；屈服强度定义参考 GB/T 228.1—2021，该标准不表示本数据集按其条件测试。")
    with st.expander("运行方式", expanded=False):
        st.code("streamlit run app.py\n\n# 重新训练（可选）\npython src/train.py\n\n# 可选本机 API\nuvicorn api:app --host 127.0.0.1 --port 8000", language="powershell")
    with st.expander("环境", expanded=False):
        st.write("依赖记录在 requirements.txt；中文字体自动探测并回退。")


PAGES = {
    "数据管理": data_management,
    "模型与评价": model_evaluation,
    "单条预测": one_prediction,
    "批量预测": batch_prediction,
    "模型解释": explanations,
    "预测记录": records_page,
    "配方对比": comparison_page,
    "关于与说明": about,
}

st.sidebar.title("铝合金屈服强度")
st.sidebar.caption("LightGBM · 特征选择 · 可追溯数据")
with st.sidebar.container(border=True):
    st.write("模型状态")
    st.caption(f"主模型：{REGISTRY['primary']}")
    st.caption(f"入模特征数：{METRICS['feature_selection']['selected_feature_count']}")
    st.caption("指标来源：metrics.json")
    st.caption(f"数据来源：{METRICS['source']}")
page_names = list(PAGES.keys())
requested_page = st.query_params.get("page", page_names[0])
page_index = page_names.index(requested_page) if requested_page in page_names else 0
page = st.sidebar.radio("页面", page_names, index=page_index)
st.query_params["page"] = page
PAGES[page]()
