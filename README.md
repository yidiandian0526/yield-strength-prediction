# 铝合金屈服强度预测系统

本项目使用 Mendeley Data V1 的 `al_data.csv` 预测铝合金屈服强度（MPa），提供已训练的 LightGBM 主模型、随机森林与 XGBoost 基线模型，以及 Streamlit 界面、纯 Python 预测服务和 FastAPI 接口。训练与测试按化学成分组合分组，固定随机种子 2024；测试集不参与训练或特征筛选。

## 环境与启动

建议使用 Python 3.10–3.12。安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

交付包已包含训练后的模型。启动网页：

```powershell
streamlit run app.py
```

可选 API：

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

接口包括 `GET /health`、`POST /predict`、`POST /explain`、`GET /records`。预测请求使用字段 `Processing` 和训练配置中记录的全部元素成分字段；成分按百分比（0–100）填写。源 CSV 的成分值以 0–1 比例记录，训练预处理乘以 100；新样本通过网页、服务或 API 输入百分比值。无真实标签的新样本只生成预测值，不计算误差。API 文档位于 `/docs`。

## 重新训练

原始数据只取自 `data/al_data.csv`。执行：

```powershell
python src/train.py
```

脚本会重建指标、模型、特征配置、选择报告、SHAP 汇总、测试预测及七张图。处理枚举先做空白归一化；剔除行索引、伸长率、抗拉强度及类别标签。成分组合分组切分为约 8:2，近零方差过滤后按相关系数 0.98 去冗余，再用 LightGBM gain 选择特征。三个模型在同一训练/测试索引和同一 MAE、RMSE、R² 指标下评价。

## 文件说明

- `models/`：已训练 LightGBM 文本模型与三模型加载文件。
- `feature_config.json`、`metrics.json`、`model_registry.json`：模型输入约定、真实指标和注册信息。
- `selection_report.csv`：特征选择前后指标；`feature_selection_details.csv`：候选字段 gain 排序和筛选阶段。
- `shap_summary.csv`：测试集全局 SHAP 汇总；网页可生成单条局部解释。
- `artifacts/figures/`：训练真实生成的七张图。网页截图保留在本地交付目录，不上传到仓库。
- `records.db`：网页或 API 首次写记录时在项目目录创建的 SQLite 数据库。
- `evidence/`：测试记录、运行说明、文献列表与引用位置覆盖表、模块映射、演示脚本、材料清单。

## 适用范围

模型仅反映所给公开数据中的模式。输入字段、单位和工艺枚举必须符合模型配置；预测值用于辅助估算，不替代材料标准试验。陌生的 Processing 类别会映射到训练时已有的类别列空间，不会凭空建立新的工艺语义。

## Streamlit Community Cloud 部署

1. 将本项目推送到 GitHub 的 `main` 分支。
2. 打开 `https://share.streamlit.io`，使用 GitHub 登录。
3. 选择仓库，Branch 填 `main`，Main file path 填 `app.py`，点击 Deploy。
4. 本项目当前不需要 Secrets；`.streamlit/secrets.toml` 已加入忽略规则，示例文件为 `.streamlit/secrets.toml.example`。如果未来代码新增 `st.secrets` 配置，在 Streamlit Cloud 的 **Settings → Secrets** 中粘贴对应 TOML 内容，不要把真实密钥提交到 GitHub。

Cloud 运行时的 `records.db` 是实例本地文件，重启或重新部署后可能被重置；模型文件和公开数据随仓库部署。
