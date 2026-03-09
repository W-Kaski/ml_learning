#!/usr/bin/env python3
"""
LSTM 时间序列预测 - 股票价格/传感器数据
=====================================

学习内容：
- LSTM 的四个门机制（输入门、遗忘门、输出门、单元门）
- 时间滑动窗口数据构造
- 归一化与反归一化处理
- 多步时间序列预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

print("=" * 80)
print("【LSTM 时间序列预测】多步股票价格预测")
print("=" * 80)
print()

# ============================================================================
# 【第一部分】LSTM 的内部机制 - 手工演示
# ============================================================================
print("【第一部分】LSTM 四个门的数学原理")
print("-" * 80)

# LSTM 在时间步 t 的四个门操作：
# f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)    # 遗忘门
# i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)    # 输入门
# C_t_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c) # 单元候选值
# C_t = f_t * C_{t-1} + i_t * C_t_tilde         # 细胞状态(记忆)
# o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)    # 输出门
# h_t = o_t * tanh(C_t)                         # 隐状态

print("""
LSTM 三个关键改进：
  ✓ 细胞状态 C_t: 独立的记忆通道，支持远距离梯度传播
  ✓ 遗忘门: 选择性地擦除旧记忆 (f_t ∈ [0,1])
  ✓ 输入门: 选择性地添加新记忆 (i_t ∈ [0,1])
  
对比 RNN:
  RNN 只有一个隐状态 h_t，易导致梯度消失/爆炸
  LSTM 有两个状态: h_t (短期) 和 C_t (长期记忆)
""")
print()

# ============================================================================
# 【第二部分】时间序列数据构造
# ============================================================================
print("【第二部分】滑动窗口数据准备")
print("-" * 80)

# 生成模拟时间序列（e.g. 股票价格）
np.random.seed(42)
torch.manual_seed(42)

# 简单的正弦+趋势+噪声
t = np.arange(0, 100, 0.5)  # 200 个时间点
trend = t * 0.1              # 上升趋势
seasonal = 10 * np.sin(2 * np.pi * t / 20)  # 周期性
noise = np.random.randn(len(t)) * 0.5       # 高斯噪声
price = 100 + trend + seasonal + noise

print(f"时间序列长度: {len(price)}")
print(f"  - 平均价格: ${price.mean():.2f}")
print(f"  - 价格范围: ${price.min():.2f} ~ ${price.max():.2f}")
print()

# 归一化
price_mean = price.mean()
price_std = price.std()
price_norm = (price - price_mean) / price_std

print(f"归一化统计:")
print(f"  - 原始: μ={price_mean:.2f}, σ={price_std:.2f}")
print(f"  - 归一化: μ={price_norm.mean():.4f}, σ={price_norm.std():.4f}")
print()

# 构造滑动窗口
lookback = 10  # 用过去 10 个时间步预测下一个
X_list, y_list = [], []

for i in range(len(price_norm) - lookback):
    X_list.append(price_norm[i:i+lookback])
    y_list.append(price_norm[i+lookback])

X_array = np.array(X_list)  # (samples, lookback)
y_array = np.array(y_list)   # (samples,)

print(f"数据集构造:")
print(f"  - Lookback window: {lookback}")
print(f"  - 样本数: {len(X_array)}")
print(f"  - 训练集 (80%): {int(0.8*len(X_array))}")
print(f"  - 测试集 (20%): {int(0.2*len(X_array))}")
print()

# 划分训练/测试
split_idx = int(0.8 * len(X_array))
X_train = torch.tensor(X_array[:split_idx]).float().unsqueeze(-1)  # (N_train, lookback, 1)
y_train = torch.tensor(y_array[:split_idx]).float()
X_test = torch.tensor(X_array[split_idx:]).float().unsqueeze(-1)
y_test = torch.tensor(y_array[split_idx:]).float()

print(f"张量形状:")
print(f"  - X_train: {X_train.shape} (batch_size, seq_len, features)")
print(f"  - y_train: {y_train.shape}")
print(f"  - X_test: {X_test.shape}")
print(f"  - y_test: {y_test.shape}")
print()

# ============================================================================
# 【第三部分】构建 LSTM 模型
# ============================================================================
print("【第三部分】LSTM 模型架构")
print("-" * 80)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后时间步的输出
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        pred = self.fc(last_hidden)  # (batch_size, output_size)
        return pred

# 模型超参数
input_size = 1
hidden_size = 16
num_layers = 2
output_size = 1

model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
params = sum(p.numel() for p in model.parameters())

print(f"模型配置:")
print(f"  - Input: {input_size}")
print(f"  - Hidden layers: {num_layers} × {hidden_size}")
print(f"  - Output: {output_size}")
print(f"  - Total parameters: {params:,}")
print()

# 显示参数分解
print(f"参数分解:")
for name, param in model.named_parameters():
    print(f"  {name:20s}: {param.shape} = {param.numel():,} params")
print()

# ============================================================================
# 【第四部分】模型训练
# ============================================================================
print("【第四部分】训练过程详解")
print("-" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"训练设置:")
print(f"  - Device: {device}")
print(f"  - Loss: MSE")
print(f"  - Optimizer: Adam (lr=0.001)")
print(f"  - Epochs: 50")
print()

# 批次训练
batch_size = 16
num_epochs = 50

train_losses = []
test_losses = []
best_test_loss = float('inf')

print("【训练进度】")
for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    
    # Shuffled minibatch
    indices = torch.randperm(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred.squeeze(), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        batch_losses.append(loss.item())
    
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred.squeeze(), y_test).item()
        test_losses.append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch
    
    # 定期输出
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}/{num_epochs}: Train Loss={train_loss:.5f}, "
              f"Test Loss={test_loss:.5f}")

print(f"\n最佳模型: Epoch {best_epoch+1}, Test Loss={best_test_loss:.5f}")
print()

# ============================================================================
# 【第五部分】预测与反归一化
# ============================================================================
print("【第五部分】模型预测结果")
print("-" * 80)

model.eval()
with torch.no_grad():
    train_pred = model(X_train).detach().cpu().numpy().flatten()
    test_pred = model(X_test).detach().cpu().numpy().flatten()
    
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

# 计算指标（归一化后的）
train_mse = ((train_pred - y_train_np) ** 2).mean()
train_mae = np.abs(train_pred - y_train_np).mean()
test_mse = ((test_pred - y_test_np) ** 2).mean()
test_mae = np.abs(test_pred - y_test_np).mean()

print(f"训练集指标:")
print(f"  - MSE: {train_mse:.5f}")
print(f"  - MAE: {train_mae:.5f}")
print(f"  - RMSE: {np.sqrt(train_mse):.5f}")
print()

print(f"测试集指标:")
print(f"  - MSE: {test_mse:.5f}")
print(f"  - MAE: {test_mae:.5f}")
print(f"  - RMSE: {np.sqrt(test_mse):.5f}")
print()

# 反归一化到原始价格空间
def denormalize(x_norm):
    return x_norm * price_std + price_mean

train_pred_orig = denormalize(train_pred)
test_pred_orig = denormalize(test_pred)
y_train_orig = denormalize(y_train_np)
y_test_orig = denormalize(y_test_np)

# 在原始价格空间计算误差
test_mape = np.abs((test_pred_orig - y_test_orig) / y_test_orig).mean() * 100

print(f"原始价格空间的指标:")
print(f"  - Test MAPE: {test_mape:.2f}%")
print(f"  - 预测价格范围: ${test_pred_orig.min():.2f} ~ ${test_pred_orig.max():.2f}")
print(f"  - 真实价格范围: ${y_test_orig.min():.2f} ~ ${y_test_orig.max():.2f}")
print()

# ============================================================================
# 【第六部分】预测示例
# ============================================================================
print("【第六部分】预测示例（前 8 个测试样本）")
print("-" * 80)

print(f"{'样本':>4} | {'真实价格':>10} | {'预测价格':>10} | {'误差 ($)':>10} | {'误差 (%)':>8}")
print("-" * 55)

for i in range(min(8, len(test_pred_orig))):
    real = y_test_orig[i]
    pred = test_pred_orig[i]
    error_val = pred - real
    error_pct = (error_val / real) * 100 if real != 0 else 0
    print(f"{i+1:4d} | ${real:9.2f} | ${pred:9.2f} | ${error_val:9.2f} | {error_pct:7.2f}%")

print()

# ============================================================================
# 【第七部分】LSTM 内部状态可视化
# ============================================================================
print("【第七部分】LSTM 隐状态与细胞状态演化")
print("-" * 80)

# 取一个测试样本，追踪 LSTM 内部状态
test_sample = X_test[0:1]  # (1, lookback, 1)

# 获取 LSTM 的中间层激活
with torch.no_grad():
    lstm_out, (h_n, c_n) = model.lstm(test_sample)
    # lstm_out: (1, lookback, hidden_size)
    # h_n: (num_layers=2, 1, hidden_size)
    # c_n: (num_layers=2, 1, hidden_size)

print(f"单个样本的 LSTM 内部状态:")
print(f"  - 输入序列长度: {test_sample.shape[1]}")
print(f"  - 每个时间步的隐状态: {lstm_out.shape}")
print(f"  - 最后时间步的隐状态 (第1层): {h_n[0].shape}")
print(f"  - 最后时间步的细胞状态 (第1层): {c_n[0].shape}")
print()

# 计算隐状态的活跃度（通过 L2 范数）
h_norms = [lstm_out[0, t].norm().item() for t in range(lookback)]
c_norms = [c_n[0, 0].norm().item()]  # 只有最终的细胞状态

print(f"隐状态 h_t 的 L2 范数随时间变化（前 {lookback} 个时间步）:")
for t, norm in enumerate(h_norms):
    bar = '█' * int(norm * 2)
    print(f"  t={t:2d}: {norm:6.3f} {bar}")

print()
print(f"最终细胞状态 C_T L2 范数: {c_norms[0]:.4f}")
print("  (细胞状态编码了整个序列的长期信息)")
print()

# ============================================================================
# 【总结】
# ============================================================================
print("=" * 80)
print("【学习总结】")
print("=" * 80)
print("""
核心概念：
  ✓ LSTM 通过细胞状态 C_t 实现长期记忆的传播
  ✓ 四个门的组合: 遗忘(what to forget) + 输入(what to add) + 输出(what to output)
  ✓ 梯度可以直接沿细胞状态反向传播，解决梯度消失问题
  
时间序列预测最佳实践：
  ✓ 数据归一化确保模型有数值稳定性
  ✓ 梯度裁剪(clip_grad_norm)防止梯度爆炸
  ✓ 测试/验证集用于提前停止(early stopping)
  ✓ 多层堆叠 LSTM 学习复杂的非线性模式
  
常用技巧：
  ✓ Attention 机制：让模型关注/忽略某些时间步
  ✓ Encoder-Decoder：多步预测（预测未来多个时间点）
  ✓ Ensemble：多个 LSTM 模型的集合预测
  
下一步：
  → 学习注意力机制 (Attention) 改进预测
  → 尝试多变量时间序列预测 (多个特征)
  → 探索 Transformer 用于序列建模
""")
print("=" * 80)
