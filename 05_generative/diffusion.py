#!/usr/bin/env python3
"""
Diffusion 教学版 - DDPM 核心流程
================================

学习内容：
- 前向加噪过程 q(x_t | x_0)
- 预测噪声 ε_theta(x_t, t)
- 反向去噪采样
- 噪声调度与时间步嵌入

说明：
- 这是一个可运行的 toy diffusion 示例，不是 Stable Diffusion 微调脚本。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


print("=" * 80)
print("【Diffusion 教学版】DDPM 核心流程")
print("=" * 80)
print()

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_toy_data(num_points=512):
    angles = torch.rand(num_points) * 2 * math.pi
    radius = 2.0 + 0.15 * torch.randn(num_points)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    return torch.stack([x, y], dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


class NoisePredictor(nn.Module):
    def __init__(self, hidden_dim=64, time_dim=32):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=1))


num_steps = 50
beta = torch.linspace(1e-4, 0.02, num_steps, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


def q_sample(x0, t, noise):
    a_bar = alpha_bar[t].unsqueeze(1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise


data = make_toy_data().to(device)
model = NoisePredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("【第一部分】数据与噪声调度")
print("-" * 80)
print(f"训练样本形状: {data.shape}")
print(f"beta[0]={beta[0].item():.6f}, beta[-1]={beta[-1].item():.6f}")
print(f"alpha_bar[0]={alpha_bar[0].item():.6f}, alpha_bar[-1]={alpha_bar[-1].item():.6f}")
print()

print("【第二部分】前向加噪直观展示")
print("-" * 80)
x0_demo = data[:3]
for t_show in [0, 10, 25, 49]:
    noise = torch.randn_like(x0_demo)
    x_t = q_sample(x0_demo, torch.full((x0_demo.size(0),), t_show, device=device, dtype=torch.long), noise)
    print(f"t={t_show:2d}: 均值={x_t.mean().item():.4f}, std={x_t.std().item():.4f}")
print()

print("【第三部分】训练噪声预测器")
print("-" * 80)
for step in range(1, 301):
    idx = torch.randint(0, data.size(0), (64,), device=device)
    x0 = data[idx]
    t = torch.randint(0, num_steps, (64,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    pred_noise = model(x_t, t)
    loss = F.mse_loss(pred_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 75 == 0:
        print(f"step {step:3d}: noise_mse={loss.item():.5f}")
print()

print("【第四部分】反向采样")
print("-" * 80)
model.eval()
with torch.no_grad():
    x = torch.randn(8, 2, device=device)
    for step in reversed(range(num_steps)):
        t = torch.full((x.size(0),), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)
        a = alpha[step]
        a_bar = alpha_bar[step]
        x = (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a)
        if step > 0:
            x = x + torch.sqrt(beta[step]) * torch.randn_like(x)
    generated = x
    radius = generated.norm(dim=1)
    for idx in range(generated.size(0)):
        print(f"样本 {idx + 1}: point={generated[idx].tolist()}, radius={radius[idx].item():.4f}")
print()

print("【第五部分】前向 vs 反向过程的理解")
print("-" * 80)
print(
    """
前向过程：
  x_0 -> x_1 -> x_2 -> ... -> x_T
  逐步加噪，最后接近纯高斯噪声

反向过程：
  x_T -> x_{T-1} -> ... -> x_0
  学一个网络预测噪声，再一步步把噪声去掉

和 GAN 的区别：
  GAN 一次性生成，训练对抗式
  Diffusion 多步去噪，训练更稳定但采样更慢
"""
)
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ Diffusion 的训练目标通常是预测加入的噪声
  ✓ 噪声调度决定了不同时间步的信息保留程度
  ✓ 采样慢但稳定，是扩散模型相对 GAN 的主要特征
  ✓ Stable Diffusion 会把扩散过程搬到潜空间并结合文本条件
"""
)
print("=" * 80)