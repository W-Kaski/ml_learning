# 机器学习系统化学习路线图 🚀

> 基于 PyTorch 2.8.0 + RTX 4080 环境
> 作者：wangp126 | 开始日期：2026-03-06

## 📚 学习路径概览

```
基础 → CNN → RNN/LSTM → Transformers → 生成模型 → 项目实战
  ↓      ↓       ↓           ↓              ↓           ↓
 张量   图像     序列       NLP/ViT       GAN/VAE    Kaggle竞赛
```

---

## 📁 目录结构

```
ml_learning/
├── 01_basics/          # 基础：张量、自动微分、优化器
├── 02_cnn/            # 卷积神经网络：图像分类、目标检测
├── 03_rnn/            # 循环神经网络：时序数据、NLP基础
├── 04_transformers/   # Transformer架构：BERT、GPT、ViT
├── 05_generative/     # 生成模型：GAN、VAE、Diffusion
├── 06_projects/       # 综合项目：端到端应用
├── datasets/          # 数据集存储
├── models/            # 训练好的模型权重
└── notebooks/         # Jupyter笔记本实验
```

---

## 🎯 第一阶段：PyTorch 基础 (1-2周)

### 目标
- 掌握张量操作
- 理解自动微分机制
- 熟悉训练循环

### 学习清单
- [x] 张量创建与操作 (`01_basics/tensor_basics.py`)
- [x] 自动求导 (`01_basics/autograd.py`)
- [x] 线性回归从零实现 (`01_basics/linear_regression.py`)
- [x] 简单神经网络 (`01_basics/simple_nn.py`)
- [x] DataLoader与数据增强 (`01_basics/data_loading.py`)

### 核心概念
```python
torch.Tensor          # 数据容器
torch.autograd        # 自动微分
nn.Module            # 模型基类
nn.functional        # 函数式API
optim.SGD/Adam       # 优化器
```

---

## 🖼️ 第二阶段：卷积神经网络 (2-3周)

### 目标
- 理解卷积、池化原理
- 实现经典CNN架构
- 掌握迁移学习

### 学习清单
- [x] **MNIST 手写数字识别** (`02_cnn/mnist_cnn.py`) ✅ 今天开始！
- [x] CIFAR-10 彩色图像分类 (`02_cnn/cifar10_resnet.py`)
- [x] 数据增强实战 (`02_cnn/augmentation.py`)
- [x] 迁移学习：预训练模型微调 (`02_cnn/transfer_learning.py`)
- [x] 目标检测入门 (YOLO) (`02_cnn/object_detection.py`)

### 经典架构进阶
```
LeNet → AlexNet → VGG → ResNet → EfficientNet → Vision Transformer
```

**实现顺序**：
1. 从零实现 LeNet (MNIST)
2. 使用 torchvision 预训练模型 (ResNet-18)
3. 自定义数据集训练
4. 模型量化与部署

---

## 📝 第三阶段：循环神经网络 (2周)

### 目标
- 处理序列数据
- 理解 RNN/LSTM/GRU 区别
- 文本生成与情感分析

### 学习清单
- [x] RNN 基础 (`03_rnn/simple_rnn.py`)
- [x] LSTM 时间序列预测 (`03_rnn/stock_prediction.py`)
- [x] 文本分类 (`03_rnn/sentiment_analysis.py`)
- [x] Seq2Seq 机器翻译 (`03_rnn/seq2seq.py`)
- [x] 注意力机制 (`03_rnn/attention.py`)

---

## 🤖 第四阶段：Transformers (3-4周)

### 目标
- 理解自注意力机制
- 使用 HuggingFace Transformers
- 微调预训练模型

### 学习清单
- [x] Transformer 从零实现 (`04_transformers/transformer_scratch.py`)
- [x] BERT 文本分类 (`04_transformers/bert_classification.py`)
- [x] GPT-2 文本生成 (`04_transformers/gpt_generation.py`)
- [x] Vision Transformer (ViT) (`04_transformers/vit_image.py`)
- [x] 多模态：CLIP (`04_transformers/clip.py`)

### 关键库
```bash
pip install transformers datasets accelerate
```

---

## 🎨 第五阶段：生成模型 (3周)

### 目标
- 理解生成式AI原理
- 实现 GAN、VAE
- 探索 Diffusion 模型

### 学习清单
- [x] VAE 图像生成 (`05_generative/vae.py`)
- [x] DCGAN 人脸生成 (`05_generative/dcgan.py`)
- [x] StyleGAN 实验 (`05_generative/stylegan.py`)
- [x] Diffusion 基础 (`05_generative/diffusion.py`)

---

## 🏆 第六阶段：项目实战 (持续)

### 学习清单
- [x] 01_image_classification：CNN 图像分类（训练 / 评估 / 推理全流程）
- [x] 02_text_classification：GRU 文本分类（训练 / 评估 / 推理全流程）
- [x] 03_multimodal_retrieval：双编码器跨模态检索（训练 / Recall@K / 检索演示）
- [x] 04_generative_lab：VAE / DCGAN / Diffusion 统一训练与采样对比实验

---

## 🛠️ 实用工具清单

### 必备库
```bash
# 核心框架
torch, torchvision, torchaudio

# 数据处理
numpy, pandas, scikit-learn

# 可视化
matplotlib, seaborn, tensorboard

# NLP
transformers, datasets, tokenizers

# 计算机视觉
opencv-python, albumentations, pillow

# 实验跟踪
wandb, mlflow
```

### 推荐资源
- **官方文档**：https://pytorch.org/docs/
- **论文复现**：https://paperswithcode.com/
- **数据集**：HuggingFace Datasets、Kaggle
- **社区**：PyTorch Forums、Reddit r/MachineLearning

---

## 📊 学习建议

### 理论与实践结合
1. **先理解原理**：看论文/教程，理解数学推导
2. **手写实现**：从零实现核心算法（如反向传播）
3. **使用框架**：用 PyTorch 高级 API 加速开发
4. **调参实验**：记录超参数影响（学习率、batch size）
5. **可视化分析**：绘制损失曲线、混淆矩阵

### 每日学习流程
```
1. 阅读代码/论文 (30min)
2. 实现/修改模型 (1-2h)
3. 训练并记录结果 (GPU 自动运行)
4. 分析实验并总结 (30min)
```

### 记录模板
每个实验创建 `experiment_log.md`：
```markdown
## 实验日期：2026-03-06
- 模型：ResNet-18
- 数据集：CIFAR-10
- 超参数：lr=0.001, batch_size=64, epochs=50
- 最佳验证准确率：92.3%
- 训练时间：23 min (GPU)
- 遇到的问题：过拟合 → 添加 Dropout
- 改进方向：尝试数据增强、学习率衰减
```

---

## 🎓 评估标准

### 初级 (1-2个月)
- ✅ 能独立实现简单 CNN 模型
- ✅ 理解训练循环和反向传播
- ✅ 会使用预训练模型

### 中级 (3-6个月)
- ✅ 能从零复现经典论文
- ✅ 熟练使用 Transformers 库
- ✅ 完成 1-2 个 Kaggle 竞赛

### 高级 (6个月+)
- ✅ 能设计自定义架构
- ✅ 理解最新研究方向
- ✅ 开源项目贡献或论文发表

---

## 🚀 快速开始

```bash
# 进入目录
cd /student/wangp126/ml_learning

# 运行第一个示例
python 02_cnn/mnist_cnn.py

# 查看训练日志
tensorboard --logdir=runs
```

---

**记住**：机器学习是实践驱动的学科，多动手、多实验！💪
