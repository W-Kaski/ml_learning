# 待实现事项 (Pending Implementation Items)

记录所有已搭架子但尚未完整实现的功能，以及已完成项目中可以改进的部分。

---

## 一、06_projects/01_image_classification — 已完成升级与真实数据训练

### 1.1 切换到真实 CIFAR-10 数据
- **当前状态**：已完成 `CIFAR-10 + ResNet-18` 完整训练与评估
- **结果**：
  - 最佳验证准确率：`78.36%`
  - 测试集准确率：`78.44%`
  - 推理示例：`airplane` 样本预测正确，置信度 `76.02%`

### 1.2 支持更强的模型
- **当前状态**：已支持 `model_name="simple_cnn" | "resnet18"`
- **已验证**：`resnet18` 的训练 / 评估 / 推理链路已全部跑通
- **可继续优化**：后续可尝试更长训练、余弦退火、预训练权重等进一步提分

---

## 二、06_projects/02_text_classification — 已完成

> **当前状态**：训练 / 评估 / 推理全流程已打通并完成验证

- [x] `config.py`：补全训练、数据、路径配置
- [x] `data_processor.py`：实现词表构建、文本编码、DataLoader 构建
- [x] `model.py`：双向 GRU 文本分类模型可运行
- [x] `train.py`：完整训练循环，保存 checkpoint，导出 `train_history.json`
- [x] `eval.py`：准确率 + Precision / Recall / F1，导出 `eval_metrics.json`
- [x] `infer.py`：支持原始文本推理，输出预测类别 + 置信度，导出 `infer_result.json`

---

## 三、06_projects/03_multimodal_retrieval — 已完成

> **当前状态**：双编码器训练、Recall@K 评估、检索演示均已完成验证

- [x] `model.py`：实现 `DualEncoder.forward()` 与 InfoNCE 对比损失
- [x] `dataset.py`：扩展为可训练的合成图文对 DataLoader
- [x] `train.py`：对比学习训练循环，记录 loss 曲线
- [x] `eval.py`：Top-k 检索准确率评估（Recall@K）
- [x] `retrieve.py`：支持文本找图与图找文检索

---

## 四、06_projects/04_generative_lab — 已完成

> **当前状态**：VAE / DCGAN / Diffusion 统一训练、采样、对比均已完成验证

- [x] `train.py`：根据 `config.model_name`（`"vae"` / `"dcgan"` / `"diffusion"`）动态加载模型并训练
- [x] `sample.py`：从训练好的模型采样生成图像，保存到 `outputs/<model_name>/samples_<checkpoint>.png`
- [x] `compare.py`：并排比较三个模型生成的样本质量，输出对比图和定量指标
- [x] `models.py`：将 VAE、DCGAN、Diffusion 统一为公共接口

---

## 五、README.md — Stage 6 部分更新

> **当前状态**：已同步为实际的 4 个子项目

- [x] 将 Stage 6 的 checklist 替换为实际的 4 个子项目：
  ```
  - [x] 01_image_classification：CNN图像分类（训练/评估/推理全流程）
  - [x] 02_text_classification：GRU文本分类（情感分析）
  - [x] 03_multimodal_retrieval：双编码器跨模态检索
  - [x] 04_generative_lab：VAE / DCGAN / Diffusion 生成模型对比实验
  ```

---

## 进度速览

| 项目 | 状态 |
|------|------|
| 01_image_classification（FakeData） | ✅ 完整实现 + 端到端验证 |
| 01_image_classification（真实 CIFAR-10） | ✅ 完整训练 + 评估完成 |
| 02_text_classification | ✅ 完整实现 + 端到端验证 |
| 03_multimodal_retrieval | ✅ 完整实现 + 端到端验证 |
| 04_generative_lab | ✅ 完整实现 + 端到端验证 |
| README Stage 6 更新 | ✅ 已同步 |
