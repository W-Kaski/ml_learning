#!/usr/bin/env python3
"""
PyTorch 基础 - 张量操作速查表
============================

这是你的第一个 PyTorch 程序！
运行这个脚本了解张量的基本操作。
"""

import torch
import numpy as np

print("=" * 70)
print("🔥 PyTorch 张量基础操作")
print("=" * 70)

# ============================================================================
# 1. 创建张量
# ============================================================================
print("\n【1. 创建张量】")

# 从列表创建
t1 = torch.tensor([1, 2, 3])
print(f"从列表创建: {t1}")

# 全零/全一张量
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
print(f"全零张量:\n{zeros}")
print(f"全一张量:\n{ones}")

# 随机张量
rand = torch.rand(2, 3)      # 均匀分布 [0, 1)
randn = torch.randn(2, 3)    # 标准正态分布
print(f"随机张量 (均匀分布):\n{rand}")
print(f"随机张量 (正态分布):\n{randn}")

# 指定范围
arange = torch.arange(0, 10, 2)  # [0, 10) 步长2
print(f"arange: {arange}")

# ============================================================================
# 2. 张量属性
# ============================================================================
print("\n【2. 张量属性】")

x = torch.randn(3, 4, 5)
print(f"形状: {x.shape}")
print(f"维度: {x.ndim}")
print(f"元素总数: {x.numel()}")
print(f"数据类型: {x.dtype}")
print(f"设备: {x.device}")

# ============================================================================
# 3. 张量运算
# ============================================================================
print("\n【3. 张量运算】")

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 逐元素运算
print(f"加法: a + b =\n{a + b}")
print(f"乘法: a * b =\n{a * b}")

# 矩阵乘法
print(f"矩阵乘法: a @ b =\n{a @ b}")

# 统计运算
x = torch.randn(3, 4)
print(f"\n张量:\n{x}")
print(f"均值: {x.mean()}")
print(f"标准差: {x.std()}")
print(f"最大值: {x.max()}")
print(f"按行求和: {x.sum(dim=1)}")

# ============================================================================
# 4. 形状操作
# ============================================================================
print("\n【4. 形状操作】")

x = torch.randn(2, 3, 4)
print(f"原始形状: {x.shape}")

# reshape
y = x.view(2, 12)  # 或 x.reshape(2, 12)
print(f"reshape 后: {y.shape}")

# 转置
z = torch.randn(3, 4)
print(f"转置前: {z.shape} → 转置后: {z.T.shape}")

# 增加维度
w = torch.randn(3, 4)
print(f"增加维度: {w.shape} → {w.unsqueeze(0).shape}")

# ============================================================================
# 5. 索引与切片
# ============================================================================
print("\n【5. 索引与切片】")

x = torch.arange(12).reshape(3, 4)
print(f"张量:\n{x}")
print(f"第一行: {x[0]}")
print(f"第一列: {x[:, 0]}")
print(f"前两行: {x[:2]}")
print(f"特定元素: {x[1, 2]}")

# ============================================================================
# 6. GPU 加速
# ============================================================================
print("\n【6. GPU 加速】")

if torch.cuda.is_available():
    device = torch.device("cuda")
    x_cpu = torch.randn(1000, 1000)
    x_gpu = x_cpu.to(device)
    
    print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"CPU 张量设备: {x_cpu.device}")
    print(f"GPU 张量设备: {x_gpu.device}")
    
    # 速度对比
    import time
    
    # CPU 计算
    start = time.time()
    _ = x_cpu @ x_cpu
    cpu_time = time.time() - start
    
    # GPU 计算
    start = time.time()
    _ = x_gpu @ x_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"\n矩阵乘法 (1000x1000):")
    print(f"  CPU 耗时: {cpu_time:.4f}s")
    print(f"  GPU 耗时: {gpu_time:.4f}s")
    print(f"  加速比: {cpu_time/gpu_time:.2f}x")
else:
    print("⚠ GPU 不可用，使用 CPU")

# ============================================================================
# 7. 与 NumPy 互转
# ============================================================================
print("\n【7. 与 NumPy 互转】")

# Tensor → NumPy
t = torch.tensor([1, 2, 3])
n = t.numpy()
print(f"Tensor: {t}, NumPy: {n}")

# NumPy → Tensor
n = np.array([4, 5, 6])
t = torch.from_numpy(n)
print(f"NumPy: {n}, Tensor: {t}")

# ============================================================================
# 8. 自动求导（重要！）
# ============================================================================
print("\n【8. 自动求导】")

# 需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

print(f"x = {x.item()}")
print(f"y = {y.item()}")

# 反向传播
y.backward()

# dy/dx = 2x + 3，当 x=2 时，dy/dx = 7
print(f"dy/dx = {x.grad.item()} (理论值: {2*x.item() + 3})")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ 基础操作完成！")
print("=" * 70)
print("\n💡 关键要点：")
print("  1. 张量是 PyTorch 的核心数据结构")
print("  2. .to(device) 用于 CPU/GPU 切换")
print("  3. requires_grad=True 启用自动求导")
print("  4. .backward() 计算梯度")
print("\n📖 下一步：运行 02_cnn/mnist_cnn.py 开始第一个实战项目！")
print("=" * 70)
