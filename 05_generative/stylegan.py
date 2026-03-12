#!/usr/bin/env python3
"""
StyleGAN 核心机制 - 教学简化版
================================

学习内容：
- Mapping Network
- AdaIN 风格调制
- Noise Injection
- Style Mixing 思想

说明：
- 这里演示的是 StyleGAN 的关键机制，不是完整的高分辨率训练脚本。
"""

import torch
import torch.nn as nn


print("=" * 80)
print("【StyleGAN 核心机制】风格调制与混合")
print("=" * 80)
print()

torch.manual_seed(42)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=16, w_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, w_dim),
        )

    def forward(self, z):
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(style_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w)
        scale, bias = style.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return scale * self.norm(x) + bias


class StyledBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.adain = AdaIN(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        x = self.conv(x)
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x = x + self.noise_weight * noise
        x = self.adain(x, w)
        return self.activation(x)


class TinyStyleGenerator(nn.Module):
    def __init__(self, z_dim=16, w_dim=16):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.constant = nn.Parameter(torch.randn(1, 32, 4, 4))
        self.block1 = StyledBlock(32, 32, w_dim)
        self.to_rgb1 = nn.Conv2d(32, 3, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.block2 = StyledBlock(32, 16, w_dim)
        self.to_rgb2 = nn.Conv2d(16, 3, 1)

    def forward(self, z1, z2=None, mix_after=1):
        batch_size = z1.size(0)
        w1 = self.mapping(z1)
        w2 = self.mapping(z2) if z2 is not None else w1

        x = self.constant.expand(batch_size, -1, -1, -1)
        x = self.block1(x, w1)
        rgb_low = self.to_rgb1(x)

        x = self.upsample(x)
        style_for_block2 = w2 if mix_after <= 1 else w1
        x = self.block2(x, style_for_block2)
        rgb_high = self.to_rgb2(x)
        return rgb_low, rgb_high, w1, w2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyStyleGenerator().to(device)
z_a = torch.randn(2, 16, device=device)
z_b = torch.randn(2, 16, device=device)

print("【第一部分】Mapping Network")
print("-" * 80)
with torch.no_grad():
    rgb_low, rgb_high, w_a, w_b = model(z_a, z_b, mix_after=1)
    print(f"z -> w 形状: {z_a.shape} -> {w_a.shape}")
    print(f"w_a 均值={w_a.mean().item():.4f}, std={w_a.std().item():.4f}")
    print(f"w_b 均值={w_b.mean().item():.4f}, std={w_b.std().item():.4f}")
print()

print("【第二部分】AdaIN 与 Noise Injection")
print("-" * 80)
print(f"低分辨率 RGB: {rgb_low.shape}")
print(f"高分辨率 RGB: {rgb_high.shape}")
print(f"高分辨率图像统计: mean={rgb_high.mean().item():.4f}, std={rgb_high.std().item():.4f}")
print()

print("【第三部分】Style Mixing 对比")
print("-" * 80)
with torch.no_grad():
    _, mixed_high, _, _ = model(z_a, z_b, mix_after=1)
    _, self_high, _, _ = model(z_a, None, mix_after=2)
    diff = (mixed_high - self_high).abs().mean().item()
    print(f"混合风格 vs 单一风格 平均差异: {diff:.4f}")
    for idx in range(2):
        print(
            f"样本 {idx + 1}: mixed_mean={mixed_high[idx].mean().item():.4f}, "
            f"self_mean={self_high[idx].mean().item():.4f}"
        )
print()

print("【第四部分】StyleGAN 设计思想")
print("-" * 80)
print(
    """
核心思想：
  1. Mapping Network 把输入 z 投影到更解耦的 w 空间
  2. AdaIN 用风格向量控制每一层特征的统计量
  3. Noise Injection 为图像局部细节引入随机性
  4. Style Mixing 让不同层负责不同尺度的语义

直观理解：
  低层更偏向姿态、布局、粗形状
  高层更偏向纹理、细节、颜色
"""
)
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ StyleGAN 的关键不是单纯“更深”，而是生成控制方式发生了变化
  ✓ 风格调制让不同层可以操控不同尺度信息
  ✓ 风格混合增强了解耦性，也便于解释潜空间
  ✓ 真实 StyleGAN 训练还会结合渐进式分辨率和正则项
"""
)
print("=" * 80)