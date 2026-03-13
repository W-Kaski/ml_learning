# 08_mlflow

目标：记录实验、管理模型、支持复盘。

建议脚本数：8

建议文件：
1. 00_mlflow_basics.py: run/params/metrics
2. 01_artifact_logging.py: 图片/模型/报告上传
3. 02_model_logging.py: mlflow.sklearn 或 pyfunc
4. 03_experiment_compare.py: 多实验对比
5. 04_registry_basics.py: 模型注册基础
6. 05_stage_transition.py: Staging/Production 切换
7. 06_mlflow_with_xgboost.py: 与 xgboost 集成
8. 07_repro_report.py: 复现实验报告

学习重点：
- 实验追踪规范
- 参数/指标命名一致性
- 结果可复现

验收标准：
- 能完整复盘一次建模实验
