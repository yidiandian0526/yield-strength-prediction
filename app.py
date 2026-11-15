from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from service.predict_service import clear_records, explain_record, list_records, load_config, predict_dataframe, predict_record

st.set_page_config(page_title="铝合金屈服强度预测", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
def detect_chinese_font() -> str:
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei"):
        if candidate in installed:
            return candidate
    return "sans-serif"


CHINESE_FONT = detect_chinese_font()
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: '{CHINESE_FONT}', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif; }}
.block-container {{ padding-top: 1.6rem; }}
h1, h2, h3 {{ color: #2C2C2A; letter-spacing: 0; }}
div[data-testid="stMetric"] {{ background: #FFF4D6; border: 1px solid #E8A317; border-radius: 6px; padding: 0.8rem; }}
button[kind="primary"] {{ background: #E8A317; color: #2C2C2A; }}
</style>
""", unsafe_allow_html=True)

CFG = load_config()
METRICS = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))


def header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def data_management() -> None:
    header("数据管理", "Mendeley Data V1 · DOI 10.17632/b6br4yk6r3.1")
    df = pd.read_csv(ROOT / "data" / "al_data.csv")
    if len(df.columns) and (str(df.columns[0]).startswith("Unnamed") or str(df.columns[0]) == ""):
        df = df.iloc[:, 1:].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", f"{len(df):,}")
    c2.metric("原始字段数", len(df.columns))
    c3.metric("目标单位", "MPa")
    st.subheader("数据预览")
    display_df = df.copy()
    for name in CFG["numeric_features"]:
        display_df[name] = pd.to_numeric(display_df[name], errors="coerce") * CFG["source_to_model_scale_factor"]
    st.caption("源 CSV 的成分以 0–1 比例存储；本页预览、训练及预测输入统一显示/使用百分比，预览值已乘 100。")
    st.dataframe(display_df.head(30), use_container_width=True, height=340)
    st.subheader("字段与处理")
    st.write("保留成分字段与 Processing；移除行索引、Elongation、Tensile Strength、class。缺少屈服强度的新样本只做预测，不计算误差。")
    st.dataframe(pd.DataFrame({"字段": CFG["numeric_features"] + [CFG["processing_field"]], "单位":["%（源 CSV 比例值×100）"] * len(CFG["numeric_features"]) + ["处理状态枚举"]}), use_container_width=True, hide_index=True)


def model_evaluation() -> None:
    header("模型与评价", "同一按成分组合分组的 8:2 测试划分，固定随机种子 2024")
    c1, c2, c3 = st.columns(3)
    c1.metric("训练样本", METRICS["train_samples"])
    c2.metric("测试样本", METRICS["test_samples"])
    c3.metric("特征数", f'{METRICS["feature_selection"]["initial_feature_count"]} → {METRICS["feature_selection"]["selected_feature_count"]}')
    st.subheader("三模型测试集指标")
    st.dataframe(pd.DataFrame(METRICS["models"]).T, use_container_width=True)
    st.subheader("模型名称与实际输入特征")
    st.write("主模型：LightGBM；对照模型：Random Forest、XGBoost。三者使用相同训练/测试划分、相同的 22 项筛选后特征和 MPa 指标。")
    st.code("、".join(CFG["selected_features"]), language=None)
    st.subheader("特征选择前后")
    st.dataframe(pd.DataFrame(METRICS["selection_comparison"]), use_container_width=True, hide_index=True)
    for filename in ["01_actual_vs_pred.png", "02_error_distribution.png", "03_feature_importance.png", "04_feature_selection_comparison.png", "07_model_comparison.png"]:
        st.image(str(ROOT / "artifacts" / "figures" / filename), use_container_width=True)


def _numeric_form(prefix: str) -> tuple[str, dict[str, float], bool]:
    with st.form(prefix):
        processing = st.selectbox("Processing 状态", CFG["processing_values"], key=prefix + "_processing")
        columns = st.columns(4)
        values: dict[str, float] = {}
        for index, name in enumerate(CFG["numeric_features"]):
            values[name] = columns[index % 4].number_input(name + " (%)", min_value=0.0, value=0.0, format="%.6f", key=prefix + "_" + name)
        submitted = st.form_submit_button("提交", type="primary")
    return processing, values, submitted


def one_prediction() -> None:
    header("单条预测", "输入各元素百分比（0–100）与处理状态，返回屈服强度预测值")
    processing, values, submitted = _numeric_form("single_prediction")
    if submitted:
        try:
            result = predict_record({**values, "Processing": processing})
            st.success(f'预测屈服强度：{result["predicted_yield_strength_MPa"]:.2f} MPa')
        except ValueError as exc:
            st.error(str(exc))


def batch_prediction() -> None:
    header("批量预测", "上传含 Processing 与成分百分比（0–100）字段的 CSV 或 Excel；无 Yield Strength 时只输出预测")
    upload = st.file_uploader("选择文件", type=["csv", "xlsx"])
    if upload:
        try:
            data = pd.read_excel(upload) if upload.name.lower().endswith("xlsx") else pd.read_csv(upload)
            result = predict_dataframe(data)
            st.dataframe(result, use_container_width=True)
            st.download_button("导出预测结果 CSV", result.to_csv(index=False).encode("utf-8-sig"), "yield_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"文件校验失败：{exc}")


def explanations() -> None:
    header("模型解释", "全局贡献来自真实测试集 SHAP；局部解释显示单条输入的特征贡献")
    st.image(str(ROOT / "artifacts" / "figures" / "05_shap_global.png"), use_container_width=True)
    st.dataframe(pd.read_csv(ROOT / "shap_summary.csv"), use_container_width=True, hide_index=True)
    st.image(str(ROOT / "artifacts" / "figures" / "06_grouped_error.png"), use_container_width=True)
    st.subheader("单条预测局部解释")
    processing, values, submitted = _numeric_form("local_explanation")
    if submitted:
        try:
            explanation = explain_record({**values, "Processing": processing})
            st.metric("屈服强度预测", f'{explanation["prediction_MPa"]:.2f} MPa')
            local = pd.DataFrame(explanation["contributions"])
            st.bar_chart(local.set_index("feature")["shap_value_MPa"])
            st.dataframe(local, use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.error(str(exc))


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
    st.dataframe(view.drop(columns=["input"], errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button("导出记录 CSV", view.to_csv(index=False).encode("utf-8-sig"), "prediction_records.csv", "text/csv")
    if st.button("清空全部记录"):
        clear_records()
        st.rerun()
    record_id = st.selectbox("记录详情", view["id"].tolist())
    with st.expander("详情", expanded=True):
        st.json(view.loc[view["id"] == record_id].iloc[0].to_dict())


def about() -> None:
    header("关于与说明", "铝合金屈服强度预测系统")
    st.markdown("""
**数据来源**  
Mendeley Data V1，DOI `10.17632/b6br4yk6r3.1`，只使用项目内 `data/al_data.csv`。

**运行方式**  
`python src/train.py` 训练；`streamlit run app.py` 启动网页；`uvicorn api:app --port 8000` 启动接口。

**适用边界**  
输入必须与训练字段和单位一致。预测结果是数据范围内的辅助估计，不能替代标准力学试验。新样本无标签时只计算预测值，不计算误差。

**环境**  
依赖记录在 `requirements.txt`，模型、特征配置、指标、图表和测试记录均随项目交付。
""")


PAGES = {"数据管理": data_management, "模型与评价": model_evaluation, "单条预测": one_prediction, "批量预测": batch_prediction, "模型解释": explanations, "预测记录": records_page, "关于与说明": about}
st.sidebar.title("铝合金屈服强度")
st.sidebar.caption("LightGBM · 特征选择 · 可追溯数据")
st.sidebar.caption(f"中文字体：{CHINESE_FONT}（自动探测，缺失时回退）")
page_names = list(PAGES.keys())
requested_page = st.query_params.get("page", page_names[0])
page_index = page_names.index(requested_page) if requested_page in page_names else 0
page = st.sidebar.radio("页面", page_names, index=page_index)
st.query_params["page"] = page
PAGES[page]()
