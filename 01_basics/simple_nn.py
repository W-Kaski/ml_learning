#!/usr/bin/env python3
"""
PyTorch 基础 - 简单神经网络
==========================

学习目标：
1. 学会定义 nn.Module
2. 理解前向传播与损失计算
3. 掌握分类任务的训练流程
"""

import torch
import torch.nn as nn

print("=" * 70)
print("🔥 PyTorch 简单神经网络（Simple NN）")
print("=" * 70)

torch.manual_seed(42)

# ============================================================================
# 1. 构造一个可分的二分类数据
# ============================================================================
print("\n【1. 构造训练数据】")

num_samples = 400
x = torch.randn(num_samples, 2)

# 决策边界：x1 + x2 > 0 记为 1，否则记为 0
y = (x[:, 0] + x[:, 1] > 0).long()

print(f"样本形状: {tuple(x.shape)}")
print(f"标签形状: {tuple(y.shape)}")
print(f"正类占比: {y.float().mean().item() * 100:.2f}%")

# ============================================================================
# 2. 定义模型
# ============================================================================
print("\n【2. 定义模型】")


class SimpleClassifier(nn.Module):
    """两层 MLP：2 -> 16 -> 2"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)


model = SimpleClassifier()
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数: {total_params}")

# ============================================================================
# 3. 训练配置
# ============================================================================
print("\n【3. 训练配置】")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("损失函数: CrossEntropyLoss")
print("优化器: Adam(lr=0.01)")

# ============================================================================
# 4. 训练循环
# ============================================================================
print("\n【4. 开始训练】")

epochs = 120
for epoch in range(1, epochs + 1):
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
        print(f"Epoch {epoch:3d}/{epochs}, Loss={loss.item():.4f}, Acc={acc:.2f}%")

# ============================================================================
# 5. 最终评估
# ============================================================================
print("\n【5. 最终评估】")

with torch.no_grad():
    final_logits = model(x)
    final_pred = final_logits.argmax(dim=1)
    final_acc = (final_pred == y).float().mean().item() * 100

print(f"最终训练准确率: {final_acc:.2f}%")

# 随机看 5 个样本预测
sample_idx = torch.tensor([0, 1, 2, 3, 4])
sample_x = x[sample_idx]
sample_y = y[sample_idx]

with torch.no_grad():
    sample_pred = model(sample_x).argmax(dim=1)

print("\n样本预测示例（真实 -> 预测）:")
for i in range(len(sample_idx)):
    print(f"  样本{i}: {sample_y[i].item()} -> {sample_pred[i].item()}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ 简单神经网络完成！")
print("=" * 70)
print("\n💡 关键要点：")
print("  1. 分类任务输出层维度 = 类别数")
print("  2. CrossEntropyLoss 输入是 logits（无需手动 softmax）")
print("  3. 训练中关注 Loss 与 Accuracy 两个指标")
print("=" * 70)
