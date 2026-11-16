# 示例输出

`sample_input.csv` 的元素成分字段均为百分比（0–100），单位为 `%`；`Processing` 为工艺状态枚举文本。
三份 CSV 均由项目随附唯一数据源 `data/al_data.csv` 抽取输入，并调用已训练模型和 `service/predict_service.py` 真实生成。
样例没有真实标签，因此只写预测值、不计算误差。`records_export.csv` 是这些演示预测的导出样例，不含本机既有 SQLite 记录；生成时数据库记录数保持不变。
