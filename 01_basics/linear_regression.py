#!/usr/bin/env python3
"""
PyTorch 基础 - 线性回归从零实现
===============================

学习目标：
1. 掌握最小二乘回归训练流程
2. 理解模型、损失函数、优化器三件套
3. 学会观察 loss 下降与参数收敛
"""

import torch

print("=" * 70)
print("🔥 PyTorch 线性回归（Linear Regression）")
print("=" * 70)

# 固定随机种子，保证每次运行结果可复现
torch.manual_seed(42)

# ============================================================================
# 1. 构造模拟数据
# ============================================================================
print("\n【1. 构造模拟数据】")

# 真实函数：y = 2.5x + 0.8 + noise
true_w = 2.5
true_b = 0.8

x = torch.linspace(-2, 2, 200).unsqueeze(1)  # shape: [200, 1]
noise = 0.2 * torch.randn_like(x)
y = true_w * x + true_b + noise

print(f"样本数: {x.shape[0]}")
print(f"特征维度: {x.shape[1]}")
print(f"目标函数: y = {true_w}x + {true_b} + noise")

# ============================================================================
# 2. 定义模型与训练组件
# ============================================================================
print("\n【2. 定义模型与训练组件】")

# 线性模型：y_hat = wx + b
model = torch.nn.Linear(in_features=1, out_features=1)

# 均方误差损失
loss_fn = torch.nn.MSELoss()

# 随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

init_w = model.weight.item()
init_b = model.bias.item()
print(f"初始参数: w={init_w:.4f}, b={init_b:.4f}")

# ============================================================================
# 3. 训练循环
# ============================================================================
print("\n【3. 开始训练】")

epochs = 200
for epoch in range(1, epochs + 1):
    # 前向传播
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs}, Loss={loss.item():.6f}")

# ============================================================================
# 4. 训练结果分析
# ============================================================================
print("\n【4. 训练结果分析】")

learned_w = model.weight.item()
learned_b = model.bias.item()

print(f"学习到的参数: w={learned_w:.4f}, b={learned_b:.4f}")
print(f"真实参数:     w={true_w:.4f}, b={true_b:.4f}")
print(f"参数误差:     |Δw|={abs(learned_w-true_w):.4f}, |Δb|={abs(learned_b-true_b):.4f}")

with torch.no_grad():
    final_loss = loss_fn(model(x), y).item()
print(f"最终 MSE: {final_loss:.6f}")

# ============================================================================
# 5. 简单预测演示
# ============================================================================
print("\n【5. 预测演示】")

test_x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
with torch.no_grad():
    pred_y = model(test_x)

for i in range(test_x.shape[0]):
    print(f"x={test_x[i].item():>4.1f} -> y_pred={pred_y[i].item():>7.3f}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ 线性回归训练完成！")
print("=" * 70)
print("\n💡 关键要点：")
print("  1. 模型: nn.Linear")
print("  2. 损失: MSELoss")
print("  3. 优化: SGD/Adam")
print("  4. 标准流程: 前向 -> 计算损失 -> 清梯度 -> 反向 -> 更新")
print("=" * 70)
