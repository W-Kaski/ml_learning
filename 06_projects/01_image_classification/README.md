# 01 Image Classification

目标：搭建一个标准图像分类项目骨架。

## 计划内容

- 数据集加载与切分
- CNN / ViT 模型切换
- 训练、评估、推理入口
- 最佳模型保存
- 输出日志与预测结果

## 目录说明

- `config.py`：训练配置
- `dataset.py`：数据集与 DataLoader
- `model.py`：模型定义
- `train.py`：训练入口
- `eval.py`：评估入口
- `infer.py`：单样本推理入口
- `outputs/`：模型和日志输出目录