#!/usr/bin/env python3
"""
PyTorch 基础 - 自动求导详解
===========================

学习目标：
1. 理解 requires_grad 的作用
2. 学会使用 backward() 反向传播
3. 掌握梯度清零与停止求导（detach/no_grad）
"""

import torch

print("=" * 70)
print("🔥 PyTorch 自动求导（Autograd）")
print("=" * 70)

# ============================================================================
# 1. 最基础：一元函数求导
# ============================================================================
print("\n【1. 一元函数求导】")

# 设置 x 需要梯度
x = torch.tensor([2.0], requires_grad=True)

# 定义函数：y = x^2 + 3x + 1
y = x ** 2 + 3 * x + 1
print(f"函数: y = x² + 3x + 1")
print(f"输入: x = {x.item()}")
print(f"前向结果: y = {y.item()}")

# 反向传播：计算 dy/dx
y.backward()
theory_grad = 2 * x.item() + 3
print(f"自动求导 dy/dx = {x.grad.item():.1f}")
print(f"理论值 dy/dx = 2x+3 = {theory_grad:.1f}")

# ============================================================================
# 2. 多变量求导
# ============================================================================
print("\n【2. 多变量求导】")

a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

# z = a*b + a^2
z = a * b + a ** 2
print(f"函数: z = a*b + a²")
print(f"输入: a={a.item()}, b={b.item()}")
print(f"前向结果: z={z.item()}")

z.backward()

# 理论梯度：dz/da = b + 2a, dz/db = a
dz_da = b.item() + 2 * a.item()
dz_db = a.item()
print(f"dz/da 自动求导: {a.grad.item():.1f} | 理论值: {dz_da:.1f}")
print(f"dz/db 自动求导: {b.grad.item():.1f} | 理论值: {dz_db:.1f}")

# ============================================================================
# 3. 梯度累积与清零
# ============================================================================
print("\n【3. 梯度累积与清零】")

w = torch.tensor([1.0], requires_grad=True)

for step in range(1, 4):
    loss = (w - 3) ** 2
    loss.backward()
    print(f"第 {step} 次 backward 后，w.grad = {w.grad.item():.1f}")

print("说明：默认梯度会累积，不会自动清零。")

# 手动清零（训练中常见）
w.grad.zero_()
print(f"清零后，w.grad = {w.grad.item():.1f}")

# ============================================================================
# 4. 向量输出的 backward（需要提供梯度）
# ============================================================================
print("\n【4. 向量输出的 backward】")

v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
u = v ** 2  # 向量输出 [1,4,9]
print(f"u = v² = {u.tolist()}")

# 向量输出必须提供“外部梯度”，这里等价于对 sum(u) 求导
u.backward(torch.ones_like(u))
print(f"du/dv = {v.grad.tolist()} (理论: [2,4,6])")

# ============================================================================
# 5. 停止求导：detach() 与 no_grad()
# ============================================================================
print("\n【5. 停止求导】")

t = torch.tensor([5.0], requires_grad=True)
p = t * 2

# detach：返回不跟踪梯度的新张量
p_detach = p.detach()
print(f"p.requires_grad = {p.requires_grad}")
print(f"p_detach.requires_grad = {p_detach.requires_grad}")

# no_grad：上下文内不构建计算图（推理阶段常用）
with torch.no_grad():
    q = p * 3
print(f"在 no_grad 中计算的 q.requires_grad = {q.requires_grad}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ Autograd 基础完成！")
print("=" * 70)
print("\n💡 关键要点：")
print("  1. requires_grad=True 才会追踪梯度")
print("  2. backward() 会把梯度累积到 .grad")
print("  3. 训练循环里记得梯度清零")
print("  4. 推理阶段用 no_grad() 省显存提速")
print("=" * 70)
