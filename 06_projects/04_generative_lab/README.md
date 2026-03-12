# 04 Generative Lab

目标：搭建一个统一比较 VAE / GAN / Diffusion 的生成模型实验台。

## 计划内容

- 统一数据接口
- 多模型切换
- 统一训练入口
- 样本保存与结果对比

## 目录说明

- `config.py`：实验配置
- `dataset.py`：统一数据输入
- `models.py`：多种生成模型入口
- `train.py`：训练入口
- `sample.py`：采样入口
- `compare.py`：结果对比入口
- `outputs/`：输出目录