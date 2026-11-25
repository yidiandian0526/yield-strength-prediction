# 铝合金屈服强度预测系统

本项目只使用 Mendeley Data V1 `al_data.csv`（DOI `10.17632/b6br4yk6r3.1`）预测屈服强度，目标单位 MPa。提供已训练的 LightGBM 主模型、随机森林和 XGBoost 对照模型，以及 Streamlit 八页界面、纯 Python 预测服务和 FastAPI 接口。交付模型可直接加载，无需用户先训练。

## 数据来源与许可

数据集名称：Aluminium alloy dataset for supervised learning；作者：N. Bhat、A. S. Barnard、N. Birbilis；版本：Mendeley Data V1；DOI：`10.17632/b6br4yk6r3.1`；许可：CC BY 4.0。请按署名许可要求注明作者、数据集名称、版本和 DOI。数据集记录的测试条件不必然符合 GB/T 228.1—2021；本项目未验证其与该标准的符合性。

## 当前实测结果

原始 1154 行；清洗过程为 1154 − 181 − 98 = 875 条。98 是去重后任一必需字段缺失/无效的行数；原始目标缺失 109 行为去重前统计。固定 seed 2024 按成分组合分组切分为训练 709 条、测试 166 条；训练/测试成分组交集为 0。特征筛选从 35 个 one-hot/成分输入列缩减到 22 个。

| 模型 | MAE (MPa) | RMSE (MPa) | R² |
| --- | ---: | ---: | ---: |
| LightGBM | 35.941530 | 48.440857 | 0.874694 |
| RandomForest | 38.994712 | 51.218459 | 0.859912 |
| XGBoost | 39.635665 | 51.638574 | 0.857605 |

输入特征与 Processing 标签映射由 `feature_config.json` 维护；最终 LightGBM 参数通过训练划分内 3 折分组交叉验证进行单变量轮换的 11 候选局部搜索，按平均 MAE 选择。本次最终仅 `min_child_samples` 从 12 改为 8，其余参数保持基准配置。稳定性结果另以 5 个种子报告，不能替代独立外部验证。

## 安装与启动

需要 Python 3.10–3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

可选 API：

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

接口：`GET /health`、`POST /predict`、`POST /explain`、`GET /records`。网页、API 和直接 Python 调用共享 `service/predict_service.py` 与 `feature_config.json`。

## 部署（Streamlit Community Cloud）

已部署在 Streamlit Community Cloud（GitHub 仓库推送后自动重新部署）。部署地址见仓库说明。部署环境为 Linux，与本地 Windows 不同，需注意两点：

- 图表中文字体：仓库自带 `assets/fonts/NotoSansSC-Regular.otf`（Noto Sans SC），`theme.py` 与 `src/train.py` 会优先注册该字体；不要删除该文件，否则部署版图表中文标题与坐标轴会显示为方框（2026-11-25 部署版复查发现的问题，已于同日修复）。
- 预测记录不持久：部署环境的 `records.db` 是全新空库，且应用重启或重新部署后记录会丢失；演示前请先在页面上跑一两次预测，长期存档请导出 CSV。

独立 CSV 预测（输入成分为 0–100 百分比，不做额外缩放；默认使用 LightGBM，输出文件默认为输入文件旁的 `<文件名>_predictions.csv`）：

```powershell
python src/predict.py --input outputs/sample_input.csv
```

可选参数：`--model random_forest` 或 `--model xgboost` 可指定模型；`--output outputs/my_predictions.csv` 可指定输出路径。输出包含 `predicted_yield_strength_MPa` 列。

## 输入约定

需要 `Processing` 和配置中列出的 25 个元素成分字段。源数据成分是 0–1 比例，训练时乘以 100；新样本输入单位为百分比（0–100）。没有真实目标值的新样本只返回预测，不计算误差。成分超出 0–100% 或未知 Processing 状态会返回可读校验错误；成分总和偏离 100% 超过 1 个百分点时返回警告并继续预测。

## 文件

- `data/al_data.csv`：唯一数据源。
- `models/`：已训练 LightGBM 文本模型、joblib 主模型副本和两项基线模型。
- `metrics.json`、`feature_config.json`、`model_registry.json`：真实指标、输入配置、模型和参数注册。
- `selection_report.csv`、`feature_selection_details.csv`：选择前后指标和训练集 gain 排序详情。
- `artifacts/test_predictions.csv`、`shap_summary.csv`：留出测试预测与 SHAP 全局贡献。
- `artifacts/figures/`：8 张正式训练分析图，含 5 种随机分组划分稳定性图；`09_threshold_sensitivity.png` 是独立附加实验图；`10_target_distribution.png` 为关于页屈服强度分布直方图的预渲染图。网页直接展示这些已渲染好的图，不再现场重画（现场重画的三模型对比图曾把 MAE/RMSE 与 R² 画在同一根轴上，导致 R² 柱子看不见，2026-11-25 已改回直接展示；关于页直方图现场重画时 Cloud 上字体注册未生效仍显示方框，同日也改回直接展示）。单条预测页的敏感性曲线因依赖用户输入无法预渲染，改为用 FontProperties 直指内置字体文件绘制。
- `artifacts/screenshots/`：当前保留 7 张网页截图，缺“配方对比”页；2026-11-25 部署版复查后，待字体修复重新部署后再补拍该页截图。
- `artifacts/sensitivity_ranges.json`：按正式 seed 2024 训练划分计算的各元素 5%–95% 扫描区间。
- `artifacts/showcase_tests.json`：补偿扫描、配方导出、外推标注和展示层一致性实测。
- `artifacts/input_validation_tests.json`、`artifacts/prediction_unit_regression.json`：输入范围/工艺校验和网页、服务、API 单位一致性实测。
- `evidence/特征选择阈值敏感性.csv`、`artifacts/figures/09_threshold_sensitivity.png`：50%/65%/80% gain 保留比例附加实验。
- `outputs/`：百分比单位输入样例、真实预测演示结果及记录导出样例；不导出本机既有预测记录。
- `evidence/`：公平性审查、调参记录、稳定性数据、字段说明、测试记录、运行说明、参考文献与交接材料。
- `environment_lock.txt`：由当前 `.venv` 导出的精确依赖版本。
- `assets/fonts/NotoSansSC-Regular.otf`：仓库自带的 Noto Sans SC 中文字体，供部署版（Linux）图表渲染使用，勿删。
- `records.db`：SQLite 本地预测记录，网页/API 保存记录时使用；部署环境上记录不持久，重启或重新部署后丢失。

## 页面功能

八页为：数据管理、模型与评价、单条预测、批量预测、模型解释、预测记录、配方对比、关于与说明。单条预测页提供典型样例填充、补偿式敏感性曲线和加入对比；配方对比页最多保存三组已计算配方，并提供 CSV 与 Markdown 导出。

## 重新训练

在项目根目录执行 `python src/train.py`。主模型调参仅使用固定主训练划分内的分组交叉验证，留出测试集不参与特征选择和选参；同一运行会重建模型、指标、SHAP、特征选择对照、图和稳定性实验。`python src/refresh_delivery.py` 可在模型已训练后重建演示 CSV 和证据摘要。

## 适用边界

模型反映所给公开数据中的模式，用于辅助估算，不替代材料标准力学试验。数据来源、字段、单位和处理标签依据见 `evidence/数据字段说明.md`；全部指标、参数与实验范围以机器生成文件为准。

