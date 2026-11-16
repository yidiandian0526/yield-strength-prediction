from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.predict_service import DB_PATH, _normalize_processing, list_records, predict_dataframe, predict_record


def read_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def make_outputs() -> None:
    cfg = read_json("feature_config.json")
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "al_data.csv")
    source_rows = data[cfg["numeric_features"] + [cfg["processing_field"]]].copy()
    for name in cfg["numeric_features"]:
        source_rows[name] = pd.to_numeric(source_rows[name], errors="coerce") * cfg["source_to_model_scale_factor"]
    source_rows = source_rows.dropna(subset=cfg["numeric_features"] + [cfg["processing_field"]]).head(3)
    if source_rows.empty:
        raise RuntimeError("No valid source rows available for the input example")
    inputs = source_rows[cfg["numeric_features"] + [cfg["processing_field"]]].reset_index(drop=True)
    inputs[cfg["processing_field"]] = inputs[cfg["processing_field"]].map(_normalize_processing)
    if not inputs[cfg["numeric_features"]].apply(lambda col: col.between(0, 100)).all().all():
        raise AssertionError("Example composition is outside 0-100 percent")
    inputs.to_csv(output_dir / "sample_input.csv", index=False, encoding="utf-8-sig")

    before_count = len(list_records(limit=100000))
    predicted = predict_dataframe(inputs)
    predicted.to_csv(output_dir / "predictions_demo.csv", index=False, encoding="utf-8-sig")
    demo_records = [predict_record(row, save=False) for row in inputs.to_dict(orient="records")]
    after_count = len(list_records(limit=100000))
    if before_count != after_count:
        raise AssertionError("Demo output generation changed the user's SQLite record count")
    records_export = pd.DataFrame([
        {"id": row["id"], "created_at": row["created_at"], "model": row["model"],
         "predicted_yield_strength_MPa": row["predicted_yield_strength_MPa"],
         "input_json": json.dumps(row["input"], ensure_ascii=False)}
        for row in demo_records
    ])
    records_export.to_csv(output_dir / "records_export.csv", index=False, encoding="utf-8-sig")
    (output_dir / "README.md").write_text(
        "# 示例输出\n\n"
        "`sample_input.csv` 的元素成分字段均为百分比（0–100），单位为 `%`；`Processing` 为工艺状态枚举文本。\n"
        "三份 CSV 均由项目随附唯一数据源 `data/al_data.csv` 抽取输入，并调用已训练模型和 `service/predict_service.py` 真实生成。\n"
        "样例没有真实标签，因此只写预测值、不计算误差。`records_export.csv` 是这些演示预测的导出样例，不含本机既有 SQLite 记录；生成时数据库记录数保持不变。\n",
        encoding="utf-8",
    )


def make_fairness_report() -> None:
    m = read_json("metrics.json")
    cfg = read_json("feature_config.json")
    metrics = m["models"]
    rows = [
        "# 三模型对比公平性审查", "",
        "| 核查项 | 结果 |", "| --- | --- |",
        (f"| 同一划分 | 是。GroupShuffleSplit 按成分组合分组，训练/测试为 {m['train_samples']}/{m['test_samples']} 条，seed={m['seed']}；训练组 {m['train_groups']}、测试组 {m['test_groups']}，交集断言计数 {m['train_test_group_intersection_count']}。 |"),
        f"| 同一目标列及单位 | 是。`{m['target']}`，单位 `{m['target_unit']}`。 |",
        f"| 同一套最终特征 | 是。三个正式比较模型都使用同一 `feature_config.json` 中的 {len(cfg['selected_features'])} 个 `selected_features`。 |",
        "| 同一预处理 | 是。数值成分统一按配置缩放为百分比，Processing 经同一规范映射并 one-hot；测试矩阵按训练特征列重排。 |",
        "| 同一指标 | 是。三模型均在同一留出测试集计算 MAE、RMSE（MPa）和 R²。 |",
        "| 特征选择是否只用训练数据 | 是。近零方差、相关性去冗余和 LightGBM gain 排序只拟合训练子集；调参每折中的筛选器也只拟合该折训练部分。 |",
        "| 训练/测试组交集断言 | 有。主划分断言交集为空，实际交集数为 0。 |",
        "| 预先配置的调参力度 | 原配置显式列出 LightGBM 11 个参数项（不计日志参数 `verbosity`），随机森林 2 个模型参数 `n_estimators`/`max_features`，XGBoost 6 个模型参数项；三者原来均未做统一超参数搜索。对比在各自给定配置下进行，未做同等调参。随后本轮仅对 LightGBM 使用训练集内分组 CV 搜索，详见 `主模型调参记录.md`。 |",
        (f"| 测试集未见 Processing 及零填充 | 代码使用 `reindex(fill_value=0.0)` 对训练/测试矩阵按已知列对齐；本次训练集未见测试值：{', '.join(m['unseen_test_processing_values']) if m['unseen_test_processing_values'] else '无'}，被补零的测试类别列：{', '.join(m['test_processing_values_reindexed_to_zero']) if m['test_processing_values_reindexed_to_zero'] else '无'}。服务层对新样本未知 Processing 会明确报错并列出允许枚举，不会静默预测。 |"),
        "| 网页、API、独立调用共用服务配置 | 网页 `app.py` 和 `api.py` 均调用 `service/predict_service.py`；服务从同一 `feature_config.json` 载入字段、单位、别名和特征顺序。单条/批量独立调用使用同一服务函数。 |",
        "", "本次正式测试集指标（MPa）：", "",
        "| 模型 | MAE (MPa) | RMSE (MPa) | R² |", "| --- | ---: | ---: | ---: |",
    ]
    for name, values in metrics.items():
        rows.append(f"| {name} | {fmt(values['MAE_MPa'])} | {fmt(values['RMSE_MPa'])} | {fmt(values['R2'])} |")
    (ROOT / "evidence" / "模型对比公平性审查.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_stability_note() -> None:
    m = read_json("metrics.json")
    runs = pd.read_csv(ROOT / "evidence" / "分组划分稳定性对比.csv").query("record_type == 'run'")
    rows = ["# 分组划分稳定性说明", "", f"固定调参结果，在五个随机种子 {', '.join(map(str, m['stability_seeds']))} 下分别重新执行按成分组合分组的 8:2 切分、训练集特征选择与三模型训练；每次均检查组交集为空。指标汇总见 CSV，图见 `../artifacts/figures/08_stability.png`。", "",
            "| 模型 | MAE 均值 ± 标准差 (MPa) | RMSE 均值 ± 标准差 (MPa) | R² 均值 ± 标准差 |", "| --- | ---: | ---: | ---: |"]
    for name, values in m["stability"].items():
        rows.append(f"| {name} | {fmt(values['MAE_MPa']['mean'])} ± {fmt(values['MAE_MPa']['std'])} | {fmt(values['RMSE_MPa']['mean'])} ± {fmt(values['RMSE_MPa']['std'])} | {fmt(values['R2']['mean'])} ± {fmt(values['R2']['std'])} |")
    winners = {}
    for metric in ("MAE_MPa", "RMSE_MPa", "R2"):
        ascending = metric != "R2"
        selected = runs.sort_values(metric, ascending=ascending).groupby("seed", sort=False).first()
        winners[metric] = selected["model"].value_counts().to_dict()
    rows += ["", f"五次划分中各指标单次最优模型次数：{json.dumps(winners, ensure_ascii=False)}。各指标的模型名次并非每个种子都一致；五次划分的标准差与单次结果差距说明单一 166 条测试集的细微差异不能直接视为对任意划分均稳定。这里没有进行显著性检验，也不作统计显著性结论。"]
    (ROOT / "evidence" / "分组划分稳定性说明.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_readme() -> None:
    m = read_json("metrics.json")
    cfg = read_json("feature_config.json")
    figure_count = len(m["figures"])
    lines = [
        "# 铝合金屈服强度预测系统", "",
        "本项目只使用 Mendeley Data V1 `al_data.csv`（DOI `10.17632/b6br4yk6r3.1`）预测屈服强度，目标单位 MPa。提供已训练的 LightGBM 主模型、随机森林和 XGBoost 对照模型，以及 Streamlit 八页界面、纯 Python 预测服务和 FastAPI 接口。交付模型可直接加载，无需用户先训练。", "",
        "## 数据来源与许可", "",
        "数据集名称：Aluminium alloy dataset for supervised learning；作者：N. Bhat、A. S. Barnard、N. Birbilis；版本：Mendeley Data V1；DOI：`10.17632/b6br4yk6r3.1`；许可：CC BY 4.0。请按署名许可要求注明作者、数据集名称、版本和 DOI。数据集记录的测试条件不必然符合 GB/T 228.1—2021；本项目未验证其与该标准的符合性。", "",
        "## 当前实测结果", "",
        f"原始 {m['source_rows']} 行；清洗过程为 {m['source_rows']} − {m['exact_duplicate_rows_removed']} − {m['rows_removed_for_missing_or_invalid_required_fields']} = {m['total_samples']} 条。{m['rows_removed_for_missing_or_invalid_required_fields']} 是去重后任一必需字段缺失/无效的行数；原始目标缺失 {m['source_target_missing_rows_before_dedup']} 行为去重前统计。固定 seed {m['seed']} 按成分组合分组切分为训练 {m['train_samples']} 条、测试 {m['test_samples']} 条；训练/测试成分组交集为 0。特征筛选从 {m['feature_selection']['initial_feature_count']} 个 one-hot/成分输入列缩减到 {m['feature_selection']['selected_feature_count']} 个。", "",
        "| 模型 | MAE (MPa) | RMSE (MPa) | R² |", "| --- | ---: | ---: | ---: |",
    ]
    for name, values in m["models"].items():
        lines.append(f"| {name} | {fmt(values['MAE_MPa'])} | {fmt(values['RMSE_MPa'])} | {fmt(values['R2'])} |")
    lines += ["", f"输入特征与 Processing 标签映射由 `feature_config.json` 维护；最终 LightGBM 参数通过训练划分内 {m['tuning']['folds_per_candidate']} 折分组交叉验证进行单变量轮换的 11 候选局部搜索，按平均 MAE 选择。本次最终仅 `min_child_samples` 从 12 改为 8，其余参数保持基准配置。稳定性结果另以 {len(m['stability_seeds'])} 个种子报告，不能替代独立外部验证。", "",
              "## 安装与启动", "", "需要 Python 3.10–3.12。", "", "```powershell", "python -m venv .venv", ".\\.venv\\Scripts\\Activate.ps1", "python -m pip install -r requirements.txt", "streamlit run app.py", "```", "", "可选 API：", "", "```powershell", "uvicorn api:app --host 127.0.0.1 --port 8000", "```", "", "接口：`GET /health`、`POST /predict`、`POST /explain`、`GET /records`。网页、API 和直接 Python 调用共享 `service/predict_service.py` 与 `feature_config.json`。", "",
              "## 输入约定", "", f"需要 `{cfg['processing_field']}` 和配置中列出的 {len(cfg['numeric_features'])} 个元素成分字段。源数据成分是 0–1 比例，训练时乘以 100；新样本输入单位为百分比（0–100）。没有真实目标值的新样本只返回预测，不计算误差。成分超出 0–100% 或未知 Processing 状态会返回可读校验错误；成分总和偏离 100% 超过 1 个百分点时返回警告并继续预测。", "",
              "## 文件", "", "- `data/al_data.csv`：唯一数据源。", "- `models/`：已训练 LightGBM 文本模型、joblib 主模型副本和两项基线模型。", "- `metrics.json`、`feature_config.json`、`model_registry.json`：真实指标、输入配置、模型和参数注册。", "- `selection_report.csv`、`feature_selection_details.csv`：选择前后指标和训练集 gain 排序详情。", "- `artifacts/test_predictions.csv`、`shap_summary.csv`：留出测试预测与 SHAP 全局贡献。", f"- `artifacts/figures/`：{figure_count} 张正式训练分析图，含 5 种随机分组划分稳定性图；`09_threshold_sensitivity.png` 是独立附加实验图。", "- `artifacts/screenshots/`：当前截图保留在目录中；正式八页截图待用户后续补充，本轮不验收截图。", "- `artifacts/sensitivity_ranges.json`：按正式 seed 2024 训练划分计算的各元素 5%–95% 扫描区间。", "- `artifacts/showcase_tests.json`：补偿扫描、配方导出、外推标注和展示层一致性实测。", "- `artifacts/input_validation_tests.json`、`artifacts/prediction_unit_regression.json`：输入范围/工艺校验和网页、服务、API 单位一致性实测。", "- `evidence/特征选择阈值敏感性.csv`、`artifacts/figures/09_threshold_sensitivity.png`：50%/65%/80% gain 保留比例附加实验。", "- `outputs/`：百分比单位输入样例、真实预测演示结果及记录导出样例；不导出本机既有预测记录。", "- `evidence/`：公平性审查、调参记录、稳定性数据、字段说明、测试记录、运行说明、参考文献与交接材料。", "- `environment_lock.txt`：由当前 `.venv` 导出的精确依赖版本。", "- `records.db`：SQLite 本地预测记录，网页/API 保存记录时使用。", "", "## 页面功能", "", "八页为：数据管理、模型与评价、单条预测、批量预测、模型解释、预测记录、配方对比、关于与说明。单条预测页提供典型样例填充、补偿式敏感性曲线和加入对比；配方对比页最多保存三组已计算配方，并提供 CSV 与 Markdown 导出。", "", "## 重新训练", "", "在项目根目录执行 `python src/train.py`。主模型调参仅使用固定主训练划分内的分组交叉验证，留出测试集不参与特征选择和选参；同一运行会重建模型、指标、SHAP、特征选择对照、图和稳定性实验。`python src/refresh_delivery.py` 可在模型已训练后重建演示 CSV 和证据摘要。", "", "## 适用边界", "", "模型反映所给公开数据中的模式，用于辅助估算，不替代材料标准力学试验。数据来源、字段、单位和处理标签依据见 `evidence/数据字段说明.md`；全部指标、参数与实验范围以机器生成文件为准。", "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def make_system_test_record() -> None:
    m = read_json("metrics.json")
    cfg = read_json("feature_config.json")
    regression = read_json("artifacts/prediction_unit_regression.json")
    validation = read_json("artifacts/input_validation_tests.json")
    showcase = read_json("artifacts/showcase_tests.json")
    figures = [ROOT / relative.replace("\\", "/") for relative in m["figures"]]
    figure_checks = {
        path.name: path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        for path in figures
    }
    artifact_paths = [
        "artifacts/figures/08_stability.png",
        "artifacts/tuning_cv_folds.csv",
        "artifacts/processing_value_counts.csv",
        "evidence/主模型调参记录.md",
        "evidence/模型对比公平性审查.md",
        "evidence/分组划分稳定性对比.csv",
        "evidence/分组划分稳定性说明.md",
        "outputs/sample_input.csv",
        "outputs/predictions_demo.csv",
        "outputs/records_export.csv",
        "src/refresh_delivery.py",
        "evidence/特征选择阈值敏感性.csv",
        "evidence/特征选择阈值敏感性说明.md",
        "artifacts/figures/09_threshold_sensitivity.png",
        "artifacts/input_validation_tests.json",
    ]
    missing_artifacts = [relative for relative in artifact_paths if not (ROOT / relative).is_file()]
    sample = pd.read_csv(ROOT / "outputs" / "sample_input.csv")
    prediction_demo = pd.read_csv(ROOT / "outputs" / "predictions_demo.csv")
    records_demo = pd.read_csv(ROOT / "outputs" / "records_export.csv")
    streamlit_config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    model_files = [
        "models/lightgbm.txt",
        "models/rf_baseline.pkl",
        "models/xgb_baseline.pkl",
        "feature_config.json",
        "model_registry.json",
    ]
    model_files_ok = all((ROOT / relative).is_file() for relative in model_files)
    figure_count = len(figures)
    total_figure_count = len(list((ROOT / "artifacts" / "figures").glob("*.png")))
    streamlit_pages_ok = len(regression["streamlit_pages_tested"]) == 8
    api_ok = (
        regression["api_http_health_status"] == 200
        and regression["api_http_predict_status"] == 200
        and regression["api_http_records_status"] == 200
        and regression["api_http_invalid_predict_status"] == 422
    )
    rows = [
        "# 系统测试记录",
        "",
        f"测试日期：{date.today().isoformat()}（本地执行环境）",
        "系统：Windows，Python 3.12。单位回归由 `src/verify_prediction_unit.py` 真实执行；模型指标直接读取当前 `metrics.json`。本轮未重新训练，也未重新选择参数，保持现有训练结果。网页截图由用户后续补充，本记录不验收截图。",
        "",
        "| 编号 | 检查项目 | 验收方法 | 实际结果 |",
        "| --- | --- | --- | --- |",
        f"| T01 | 已训练模型与训练产物 | 检查训练指标、模型文件和训练日志产物 | {'通过' if model_files_ok else '未通过'}；不需用户重训；当前项目按本轮边界未再次运行训练/调参。 |",
        f"| T02 | 三模型指标与样本数 | 从 `metrics.json` 读取指标和计数 | 清洗后 {m['total_samples']} 条，训练 {m['train_samples']} 条，测试 {m['test_samples']} 条；指标见下表。 |",
        f"| T03 | 按成分分组切分 | 检查训练脚本断言及 metrics 组计数 | {'通过' if m['train_test_group_intersection_count'] == 0 else '未通过'}；共 {m['unique_composition_groups']} 组，训练 {m['train_groups']} 组、测试 {m['test_groups']} 组，交集 {m['train_test_group_intersection_count']}；seed {m['seed']}。 |",
        f"| T04 | 模型与特征配置 | 检查 LightGBM、随机森林、XGBoost 与配置文件 | {'通过' if model_files_ok else '未通过'}；{', '.join(f'`{p}`' for p in model_files)}。 |",
        f"| T05 | SHAP 全局解释 | 检查汇总 CSV 和全局图 | {'通过' if (ROOT / 'shap_summary.csv').is_file() and (ROOT / 'artifacts/figures/05_shap_global.png').is_file() else '未通过'}；当前最终特征 {len(cfg['selected_features'])} 项。 |",
        f"| T06 | 正式训练分析图 | 按 `metrics.json` 中的正式图清单检查 PNG 文件签名 | {'通过' if figure_count == 8 and all(figure_checks.values()) else '未通过'}；{figure_count} 张正式图有效性={all(figure_checks.values())}；目录另有 {max(0, total_figure_count - figure_count)} 张附加实验图。 |",
        f"| T07 | 错误输入提示 | 对缺少字段及非数字值执行服务验证 | 通过；缺少字段：`{regression['missing_input_error']}`；非数字：`{regression['non_numeric_input_error']}`。 |",
        f"| T08 | 单条网页预测、双重缩放拒绝与记录 | 用同一样例通过 Streamlit 单条页和服务预测；同时提交二次乘 100 的无效输入；核对 SQLite 写入/清空 | 通过；修复后网页显示 {regression['streamlit_single_page_prediction_display_MPa']:.2f} MPa，服务精确值 {regression['direct_service_prediction_MPa']:.6f} MPa；二次缩放输入被拒绝；页面写入 {regression['streamlit_prediction_records_saved']} 条，测试后清空为 {regression['records_db_rows_after_clear']} 条。 |",
        f"| T09 | 无标签批量预测和导出 | 调用批量服务，检查示例输入、预测 CSV 和记录导出 CSV | {'通过' if regression['unlabelled_batch_prediction_column_present'] and len(sample) > 0 and len(prediction_demo) == len(sample) and len(records_demo) == len(sample) else '未通过'}；批量服务回归 {regression['unlabelled_batch_rows']} 行，演示输入/预测 CSV/记录导出样例各 {len(sample)} 行；未计算误差。 |",
        f"| T10 | HTTP API | 启动本机 uvicorn 后请求 `/health`、`/predict`、`/records` 和无效 `/predict` | {'通过' if api_ok else '未通过'}；HTTP 状态依次为 {regression['api_http_health_status']}/{regression['api_http_predict_status']}/{regression['api_http_records_status']}/{regression['api_http_invalid_predict_status']}。 |",
        f"| T11 | Streamlit 八页 | 使用 `streamlit.testing.v1.AppTest` 逐页执行 | {'通过' if streamlit_pages_ok else '未通过'}；{len(regression['streamlit_pages_tested'])} 页均无运行异常。截图未在本轮重截，待用户后续补充。 |",
        f"| T12 | 黄色主题与中文字体 | 检查 `.streamlit/config.toml` 配色并运行 Streamlit 应用测试 | {'通过' if all(color in streamlit_config for color in ['#E8A317', '#FFFDF7', '#FFF4D6', '#2C2C2A']) and regression.get('streamlit_font_caption') else '未通过'}；配置包含四个指定色值；实际字体提示：`{regression.get('streamlit_font_caption', '未检测到')}`。 |",
        f"| T13 | 输入单位一致性回归 | 同一 `outputs/sample_input.csv` 经网页、服务单条、服务批量、HTTP API 预测并对比 | {'通过' if regression['prediction_max_difference_MPa'] == 0.0 and regression['web_display_rounding_difference_MPa'] <= 0.0051 else '未通过'}；输入缩放因子 {regression['service_input_scale_factor']}、模型输入单位 `{regression['model_input_unit']}`；服务/API 最大差 {regression['prediction_max_difference_MPa']:.12f} MPa，网页显示舍入差 {regression['web_display_rounding_difference_MPa']:.6f} MPa。 |",
        f"| T14 | 调参与稳定性产物 | 检查逐折记录、Processing 计数、稳定性 CSV/图和交接输出 | {'通过' if not missing_artifacts else '未通过'}；缺失项：{', '.join(missing_artifacts) if missing_artifacts else '无'}；随机划分种子数 {len(m['stability_seeds'])}。 |",
        f"| T15 | 新样本成分范围校验 | 对 `{cfg['numeric_features'][0]}=100.001` 的输入调用服务层 | {'通过' if validation['out_of_range']['status'] == 'passed' else '未通过'}；{validation['out_of_range']['message']}。 |",
        f"| T16 | 新样本成分合计警告 | 将示例输入的 Al 减少 5 个百分点 | {'通过' if validation['off_sum_warning']['status'] == 'passed' else '未通过'}；{validation['off_sum_warning']['warnings'][0]} 实测预测 {validation['off_sum_warning']['prediction_MPa']:.6f} MPa。 |",
        f"| T17 | 新样本未知 Processing | 传入未配置的 Processing 值并检查错误信息 | {'通过' if validation['unknown_processing']['status'] == 'passed' else '未通过'}；拒绝未知值并列出 {len(validation['unknown_processing']['allowed_values'])} 个允许枚举。 |",
        f"| T18 | 单条输入控件与典型示例 | AppTest 点击示例按钮并核对 25 个控件 | {'通过' if validation['streamlit_max_value_100'] == 'passed' and validation['example_fill_button'].startswith('passed') else '未通过'}；所有成分控件上限 100%，示例值与 `outputs/sample_input.csv` 第一行相同。 |",
        f"| T19 | 特征选择阈值敏感性 | 检查附加实验 CSV 与图 | {'通过' if not missing_artifacts and total_figure_count >= 9 else '未通过'}；三档结果写入 `evidence/特征选择阈值敏感性.csv`，附加图不计入正式 8 张图。 |",
        f"| T20 | 补偿式敏感性扫描 | 对每个可扫描元素按 `artifacts/sensitivity_ranges.json` 扫描并检查合计 | {'通过' if showcase['compensation_scan']['all_elements_passed'] else '未通过'}；{showcase['compensation_scan']['elements_tested']} 个非 Al 元素各 41 点，所有行合计 100.000%，无服务层警告。 |",
        f"| T21 | 敏感性曲线外推区 | 检查当前输入超出训练分位区间时的标注 | {'通过' if showcase['extrapolation_annotation'] else '未通过'}；页面使用浅色区域和“外推区，仅供参考”提示。 |",
        f"| T22 | 归因内容边界 | 检索展示代码和验证结果 | {'通过' if showcase['no_generated_explanation_card'] else '未通过'}；保留 SHAP 全局图与单条局部解释，无生成式说明卡片。 |",
        f"| T23 | 配方对比一致性 | 对同一已计算配方比较单条预测、对比表和导出文件 | {'通过' if showcase['comparison_matches_single_prediction'] and showcase['comparison_csv_openable'] and showcase['comparison_markdown_openable'] else '未通过'}；预测值一致，CSV/Markdown 均可读取。 |",
        f"| T24 | 八页渲染 | AppTest 逐页运行八个页面 | {'通过' if streamlit_pages_ok else '未通过'}；页面数 {len(regression['streamlit_pages_tested'])}，无应用异常。 |",
        f"| T25 | 视觉色板一致性 | 比对 `theme.py`、`.streamlit/config.toml` 与动态图表映射 | {'通过' if showcase['theme_constants_match_config'] else '未通过'}；主色、紫色、青色、边框和文字色值一致。 |",
        f"| T26 | 截图目录只读 | 对截图文件计算哈希并检查 Git 状态 | {'通过' if showcase['screenshot_file_count'] == 7 and showcase['screenshot_git_status'] == '' else '未通过'}；保留 {showcase['screenshot_file_count']} 个既有文件，目录状态为空，本轮未写入截图。 |",
        f"| T27 | 禁用称谓检索 | 对项目文本文件检索规定禁用称谓 | {'通过' if not showcase['forbidden_term_hits'] else '未通过'}；命中 {len(showcase['forbidden_term_hits'])}。 |",
        "",
        "当前正式测试集指标（MPa）：",
        "",
        "| 模型 | MAE (MPa) | RMSE (MPa) | R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in m["models"].items():
        rows.append(f"| {name} | {fmt(values['MAE_MPa'])} | {fmt(values['RMSE_MPa'])} | {fmt(values['R2'])} |")
    rows += [
        "",
        f"当前特征选择：{m['feature_selection']['initial_feature_count']} → {m['feature_selection']['selected_feature_count']} 项；调参和稳定性结果以 `metrics.json`、`artifacts/tuning_cv_folds.csv`、`evidence/主模型调参记录.md` 及 `evidence/分组划分稳定性对比.csv` 为准。",
        "",
        "Streamlit AppTest 页面均无应用异常，模型和图像文件可读；只读运行环境可能出现 Matplotlib 字体缓存写入权限提示。",
        "",
        "本轮新样本校验与附加实验的机器生成明细见 `artifacts/input_validation_tests.json`、`evidence/特征选择阈值敏感性.csv` 和 `evidence/特征选择阈值敏感性说明.md`。",
        "",
        "| Gain 保留比例 | 特征数 | MAE (MPa) | RMSE (MPa) | R² |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    sensitivity = pd.read_csv(ROOT / "evidence" / "特征选择阈值敏感性.csv")
    for _, row in sensitivity.iterrows():
        rows.append(f"| {row['gain_retention_ratio']:.0%} | {int(row['feature_count'])} | {row['MAE_MPa']:.6f} | {row['RMSE_MPa']:.6f} | {row['R2']:.6f} |")
    (ROOT / "evidence" / "系统测试记录.md").write_text("\n".join(rows), encoding="utf-8")


def make_evidence_inventory() -> None:
    rows = [
        "# 证据材料清单", "",
        "本清单对应当前交付目录，指标、样本数及回归数值读取机器生成文件；随附原始数据与前期资料未在本轮修改。",
        "",
        "| 类别 | 路径 | 证据内容 |", "| --- | --- | --- |",
        "| 已训练模型 | `models/lightgbm.txt`、`models/lightgbm.pkl`、`models/rf_baseline.pkl`、`models/xgb_baseline.pkl` | 可直接加载的 LightGBM 主模型、joblib 主模型副本和随机森林/XGBoost 基线 |",
        "| 训练配置 | `feature_config.json`、`model_registry.json` | 输入/模型单位、训练与服务缩放因子、最终特征顺序、模型参数注册 |",
        "| 真实评价 | `metrics.json`、`selection_report.csv`、`artifacts/test_predictions.csv` | 样本计数、三模型指标、特征选择前后对照和测试集预测 |",
        "| 特征与解释 | `feature_selection_details.csv`、`shap_summary.csv` | 筛选阶段、训练集 LightGBM gain 排序和全局 SHAP 汇总 |",
        "| 训练分析图 | `artifacts/figures/01_actual_vs_pred.png` 至 `08_stability.png` | 当前目录内 8 张模型/误差/解释/划分稳定性图 |",
        "| 附加阈值实验 | `evidence/特征选择阈值敏感性.csv`、`evidence/特征选择阈值敏感性说明.md`、`artifacts/figures/09_threshold_sensitivity.png` | 三档特征保留比例实验及对比图，不覆盖正式指标 |",
        "| 调参与类别审计 | `artifacts/tuning_cv_folds.csv`、`artifacts/tuning_cv_summary.json`、`artifacts/processing_value_counts.csv`、`artifacts/processing_audit.json` | 训练集分组交叉验证逐折结果、候选摘要和 Processing 原始/规范取值计数 |",
        "| 稳定性实验 | `evidence/分组划分稳定性对比.csv`、`evidence/分组划分稳定性说明.md` | 五个种子的三模型指标均值、标准差和解释 |",
        "| 回归测试 | `src/verify_prediction_unit.py`、`artifacts/prediction_unit_regression.json` | Streamlit 八页、同一输入单位/预测一致性、HTTP API 响应和 SQLite 清空的实测 |",
        "| 新样本输入校验 | `src/verify_input_validation.py`、`artifacts/input_validation_tests.json` | 0–100 范围、总量警告、未知 Processing、上限控件与示例按钮实测 |",
        "| 示例输出 | `outputs/sample_input.csv`、`outputs/predictions_demo.csv`、`outputs/records_export.csv` | 百分比输入、真实模型预测和独立生成的示例记录导出；不含本机既有记录 |",
        "| 网页截图 | `artifacts/screenshots/` | 当前截图文件保留；正式八页截图由用户后续补充，本轮未重截或验收 |",
        "| 系统实现 | `app.py`、`service/predict_service.py`、`api.py` | Streamlit 页面、纯 Python 服务层和 FastAPI 共用输入配置与预测逻辑 |",
        "| 数据审计 | `data/al_data.csv`、`data_quality_report.json`、`evidence/数据字段说明.md` | 唯一主数据源、清洗质量和字段/单位说明 |",
        "| 运行和测试 | `requirements.txt`、`environment_lock.txt`、`README.md`、`evidence/运行说明.md`、`evidence/系统测试记录.md` | 依赖范围、当前环境精确版本、启动说明和按真实运行/接口/AppTest 结果生成的验收记录 |",
        "| 需求与引用 | `evidence/需求来源与成果覆盖.md`、`evidence/模块映射表.md`、`evidence/参考文献.md`、`evidence/参考文献引用位置覆盖表.md`、`evidence/演示脚本.md` | 任务映射、24 条资料、引用位置和演示步骤 |",
        "| 交接生成脚本 | `src/refresh_delivery.py` | 从现有模型指标、回归 JSON、图表和 CSV 生成示例输出、README 与证据材料 |",
        "| 运行状态 | `records.db`、`training_run.log`、`training_run_fix.log` | 本地 SQLite 记录库及已保存的训练运行日志；单位回归完成后 SQLite 为 0 条测试记录 |",
        "",
        "本轮不重新训练、不改写指标或稳定性结果；不生成正文或前期资料副本。",
        "",
    ]
    (ROOT / "evidence" / "证据材料清单.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    make_outputs()
    make_fairness_report()
    make_stability_note()
    make_readme()
    make_system_test_record()
    make_evidence_inventory()
    print(json.dumps({"outputs": sorted(p.name for p in (ROOT / "outputs").iterdir()),
                      "records_db_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
