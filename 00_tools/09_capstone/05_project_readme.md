# 09 Capstone: Credit Risk Classification

## 项目目标
- 串联数据处理、建模、评估可视化、服务化、实验追踪全链路。
- 提供可复现、一键可跑的最小交付项目。

## 项目结构
- `00_data_pipeline.py`: 生成原始数据、清洗、特征工程
- `01_train_model.py`: 训练 XGBoost 分类模型并导出
- `02_eval_and_plots.py`: 输出 ROC/PR/混淆矩阵评估图
- `03_service_fastapi.py`: 推理 API（含鉴权和自测）
- `04_mlflow_tracking.py`: MLflow 记录参数、指标和模型

## 一键复现步骤
1. 数据处理
```bash
python3 00_data_pipeline.py
```
2. 训练模型
```bash
python3 01_train_model.py
```
3. 评估与图表
```bash
python3 02_eval_and_plots.py
```
4. 运行推理服务（可选）
```bash
python3 -m uvicorn 03_service_fastapi:app --host 0.0.0.0 --port 8020
```
5. 记录 MLflow 结果
```bash
python3 04_mlflow_tracking.py
```

## 快速接口测试
```bash
curl -X POST "http://127.0.0.1:8020/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer capstone-token" \
  -d '{
    "age": 39,
    "income": 90000,
    "loan_amount": 45000,
    "credit_utilization": 0.42,
    "late_payments_12m": 1,
    "region": "East",
    "channel": "Online"
  }'
```

## 主要产物
- 数据: `data/credit_risk_raw.csv`, `data/credit_risk_processed.csv`
- 模型: `models/credit_risk_xgb.joblib`
- 评估: `outputs/evaluation_dashboard.png`, `outputs/classification_report.txt`
- 追踪: `mlruns_data/mlflow.db` 与 MLflow run artifacts

## 结果解读建议
- 先看 `AUC` 与 `F1` 平衡区间是否满足业务目标。
- 结合混淆矩阵重点检查假阴性（高风险漏判）占比。
- 若要偏保守策略，可在服务中下调阈值（当前 0.5）。
