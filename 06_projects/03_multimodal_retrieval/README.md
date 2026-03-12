# 03 Multimodal Retrieval

目标：搭建一个图文对齐与检索项目骨架。

## 计划内容

- 图像编码器与文本编码器
- 共享嵌入空间
- 对比学习训练
- Top-k 检索结果输出

## 目录说明

- `config.py`：训练配置
- `dataset.py`：图文配对数据
- `model.py`：双塔检索模型
- `train.py`：训练入口
- `eval.py`：检索评估入口
- `retrieve.py`：单查询检索入口
- `outputs/`：输出目录