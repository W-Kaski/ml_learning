#!/usr/bin/env python3
"""
RNN 基础 - 循环神经网络入门
==========================

学习内容：
- RNN 的基本结构与前向传播
- 序列输入与隐状态更新
- 对比 RNN vs LSTM vs GRU
- 梯度流与截断反向传播(TBPTT)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("【RNN 基础教学】循环神经网络核心概念")
print("=" * 80)
print()

# ============================================================================
# 【第一部分】RNN 手工计算 - 理解内部机制
# ============================================================================
print("【第一部分】RNN 手工前向传播 - 验证数学")
print("-" * 80)

# 极简 RNN 单元：h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
batch_size = 2
input_size = 3
hidden_size = 4
seq_len = 3

# 手工初始化权重（小值防止梯度爆炸）
np.random.seed(42)
torch.manual_seed(42)

W_ih = torch.randn(hidden_size, input_size) * 0.01  # 输入→隐层
W_hh = torch.randn(hidden_size, hidden_size) * 0.01  # 隐层→隐层（循环权重）
b_h = torch.zeros(hidden_size)

# 生成 seq_len 个时间步的输入
x_seq = torch.randn(seq_len, batch_size, input_size)  # (T, B, input_size)

print(f"输入形状: x_seq = {x_seq.shape}")
print(f"权重形状: W_ih = {W_ih.shape}, W_hh = {W_hh.shape}")
print()

# 手工迭代计算隐状态
h_t = torch.zeros(batch_size, hidden_size)  # 初始隐状态
h_seq = []

print("【时间步迭代过程】")
for t in range(seq_len):
    x_t = x_seq[t]  # (batch_size, input_size)
    # 标准 RNN 公式: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    h_t = torch.tanh(x_t @ W_ih.t() + h_t @ W_hh.t() + b_h)
    h_seq.append(h_t.clone())
    print(f"  t={t}: x_t shape={x_t.shape} → h_t shape={h_t.shape}, "
          f"h_t[0] norm={h_t[0].norm().item():.4f}")

h_manual = torch.stack(h_seq)  # (T, B, hidden_size)
print(f"\n手工计算结果: h_manual = {h_manual.shape}")
print(f"最后隐状态 h_T[0] = {h_manual[-1][0][:3]}...")  # 显示部分值
print()

# ============================================================================
# 【第二部分】PyTorch RNN 层 - 快速对比
# ============================================================================
print("【第二部分】PyTorch 内置 RNN 与手工计算对比")
print("-" * 80)

# 使用 PyTorch 的 RNNCell 精确复现
rnn_cell = nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
# 复制手工权重到 RNNCell（验证两者一致性）
rnn_cell.weight_ih.data = W_ih
rnn_cell.weight_hh.data = W_hh
rnn_cell.bias_ih.data = b_h
rnn_cell.bias_hh.data.zero_()

h_t_pytorch = torch.zeros(batch_size, hidden_size)
h_seq_pytorch = []

for t in range(seq_len):
    x_t = x_seq[t]
    h_t_pytorch = rnn_cell(x_t, h_t_pytorch)
    h_seq_pytorch.append(h_t_pytorch.clone())

h_pytorch = torch.stack(h_seq_pytorch)

# 比较两者差异
diff = (h_manual - h_pytorch).abs().max().item()
print(f"手工 vs PyTorch 最大差异: {diff:.2e}")
if diff < 1e-5:
    print("✓ 完全一致！验证了 RNN 数学原理\n")
else:
    print(f"⚠ 存在差异（通常由于数值精度）\n")

# ============================================================================
# 【第三部分】RNN 序列到序列 - 分类任务
# ============================================================================
print("【第三部分】RNN 用于序列分类任务")
print("-" * 80)

# 模拟序列数据：长度=10, 特征=5
seq_length = 10
feature_size = 5
hidden_dim = 8
num_classes = 3

class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, h_n = self.rnn(x)  # out: (B, T, hidden), h_n: (1, B, hidden)
        # 取最后一个时间步的隐状态作为序列表示
        last_hidden = h_n.squeeze(0)  # (B, hidden)
        logits = self.fc(last_hidden)  # (B, num_classes)
        return logits

model = SimpleRNNClassifier(feature_size, hidden_dim, num_classes)
params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {params:,}")
print(f"架构: Embedding(seq_len={seq_length}) → RNN(hidden={hidden_dim}) → FC({num_classes})")

# 生成随机序列数据
batch = 4
X = torch.randn(batch, seq_length, feature_size)
y = torch.randint(0, num_classes, (batch,))

print(f"\n输入批次: X shape={X.shape}, y shape={y.shape}")

# 前向传播
logits = model(X)
print(f"输出 logits: shape={logits.shape}")
print(f"logits[0] = {logits[0]}")

# 计算损失
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, y)
print(f"交叉熵损失: {loss.item():.4f}")
print()

# ============================================================================
# 【第四部分】梯度流验证 - 检查梯度是否能反向传播
# ============================================================================
print("【第四部分】梯度流与反向传播")
print("-" * 80)

model.zero_grad()
logits = model(X)
loss = criterion(logits, y)
loss.backward()

# 检查梯度
print("【梯度统计】")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.data.norm().item()
        print(f"  {name:20s}: param_norm={param_norm:8.4f}, grad_norm={grad_norm:8.4f}, "
              f"ratio={grad_norm/param_norm if param_norm > 0 else 0:8.4f}")
print()

# ============================================================================
# 【第五部分】RNN vs LSTM vs GRU 对比
# ============================================================================
print("【第五部分】三种 RNN 变体对比")
print("-" * 80)

models_dict = {
    'RNN': nn.RNN(feature_size, hidden_dim, batch_first=True),
    'LSTM': nn.LSTM(feature_size, hidden_dim, batch_first=True),
    'GRU': nn.GRU(feature_size, hidden_dim, batch_first=True),
}

print(f"\n所有模型输入: X shape={X.shape}")
print()

for name, model_variant in models_dict.items():
    try:
        if name == 'LSTM':
            out, (h_n, c_n) = model_variant(X)
        else:
            out, h_n = model_variant(X)
        
        params = sum(p.numel() for p in model_variant.parameters())
        print(f"{name:6s}:")
        print(f"  输出 shape: {out.shape}")
        print(f"  参数量: {params:,}")
        
        # 计算参数倍数（相对 RNN）
        if name == 'RNN':
            rnn_params = params
        else:
            ratio = params / rnn_params
            print(f"  参数倍数 (vs RNN): {ratio:.1f}x")
        print()
    except Exception as e:
        print(f"{name} 处理失败: {e}\n")

# ============================================================================
# 【第六部分】序列截断反向传播(TBPTT) - 演示
# ============================================================================
print("【第六部分】截断反向传播 (Truncated BPTT)")
print("-" * 80)

# 模拟长序列
long_seq_len = 100
X_long = torch.randn(2, long_seq_len, feature_size)

# 不截断 vs 截断
truncate_step = 10

print(f"长序列长度: {long_seq_len}")
print(f"截断步长: {truncate_step}")
print()

# 完整 BPTT
model_full = nn.RNN(feature_size, hidden_dim, batch_first=True)
out_full, _ = model_full(X_long)
loss_full = out_full.sum()  # 虚拟损失
loss_full.backward()

grad_norm_full = sum(p.grad.norm() ** 2 for p in model_full.parameters() 
                     if p.grad is not None) ** 0.5
print(f"完整 BPTT:")
print(f"  梯度范数: {grad_norm_full:.4f}")
print()

# 截断 BPTT
model_trunc = nn.RNN(feature_size, hidden_dim, batch_first=True)
model_trunc.load_state_dict(model_full.state_dict())

total_loss = 0
h_t = None
for chunk_idx in range(0, long_seq_len, truncate_step):
    chunk = X_long[:, chunk_idx:chunk_idx+truncate_step, :]
    
    # 重要：截断隐状态梯度（阻止太长的反向传播）
    if h_t is not None:
        h_t = h_t.detach()
    
    out, h_t = model_trunc(chunk, h_t)
    total_loss += out.sum()

model_trunc.zero_grad()
total_loss.backward()

grad_norm_trunc = sum(p.grad.norm() ** 2 for p in model_trunc.parameters() 
                      if p.grad is not None) ** 0.5
print(f"截断 BPTT (step={truncate_step}):")
print(f"  梯度范数: {grad_norm_trunc:.4f}")
print(f"  计算复杂度: O(T/{truncate_step}) ≈ {long_seq_len // truncate_step} 步")
print()

# ============================================================================
# 【第七部分】隐状态演化可视化
# ============================================================================
print("【第七部分】隐状态在时间上的演化")
print("-" * 80)

# 使用单个示例追踪隐状态
simple_rnn = nn.RNN(feature_size, 16, batch_first=True)
X_viz = torch.randn(1, 20, feature_size)
out_viz, _ = simple_rnn(X_viz)  # (1, 20, 16)

# 计算隐状态的 L2 范数随时间的变化
h_norms = [out_viz[0, t].norm().item() for t in range(20)]

print("隐状态范数变化（前20个时间步）:")
for t, norm in enumerate(h_norms):
    bar = '█' * int(norm * 5)
    print(f"  t={t:2d}: {norm:6.3f} {bar}")
print()

# ============================================================================
# 【总结】
# ============================================================================
print("=" * 80)
print("【学习总结】")
print("=" * 80)
print("""
核心概念：
  ✓ RNN 通过隐状态 h_t 捕捉序列中的长期依赖
  ✓ 标准 RNN 公式: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
  ✓ LSTM/GRU 通过门机制解决梯度消失/爆炸问题
  
关键技术：
  ✓ 截断反向传播(TBPTT)用于长序列训练（O(T/τ) 复杂度）
  ✓ 梯度裁剪(Gradient Clipping)防止梯度爆炸
  ✓ 序列表示通常取最后时间步的隐状态
  
下一步：
  → 学习 LSTM 的门机制原理
  → 尝试时间序列预测任务 (stock_prediction.py)
  → 实现双向 RNN (BiRNN) 用于文本分类
""")
print("=" * 80)
