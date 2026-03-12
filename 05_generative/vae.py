#!/usr/bin/env python3
"""
VAE 图像生成 - 变分自编码器教学版
==================================

学习内容：
- 编码器输出均值与方差
- 重参数化技巧
- 重构损失 + KL 散度
- 潜空间采样与插值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


print("=" * 80)
print("【VAE 图像生成】重构与潜变量建模")
print("=" * 80)
print()

torch.manual_seed(42)


class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kl, recon_loss, kl


transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FakeData(size=256, image_size=(1, 28, 28), num_classes=10, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("【第一部分】数据与模型")
print("-" * 80)
batch_images, _ = next(iter(loader))
print(f"输入图像形状: {batch_images.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print()

print("【第二部分】训练")
print("-" * 80)
for epoch in range(1, 6):
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    sample_mu = None
    for images, _ in loader:
        images = images.to(device)
        recon, mu, logvar, z = model(images)
        loss, recon_loss, kl = vae_loss(recon, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_recon += recon_loss.item() * images.size(0)
        total_kl += kl.item() * images.size(0)
        sample_mu = mu.detach()
    num_samples = len(loader.dataset)
    print(
        f"Epoch {epoch}: loss={total_loss / num_samples:.4f}, "
        f"recon={total_recon / num_samples:.4f}, kl={total_kl / num_samples:.4f}, "
        f"mu_std={sample_mu.std().item():.4f}"
    )
print()

print("【第三部分】重构效果")
print("-" * 80)
model.eval()
with torch.no_grad():
    images, _ = next(iter(loader))
    images = images[:4].to(device)
    recon, mu, logvar, z = model(images)
    mse = F.mse_loss(recon, images).item()
    print(f"4 张样本的重构 MSE: {mse:.6f}")
    for idx in range(4):
        print(
            f"样本 {idx + 1}: 原图均值={images[idx].mean().item():.4f}, "
            f"重构均值={recon[idx].mean().item():.4f}, z_norm={z[idx].norm().item():.4f}"
        )
print()

print("【第四部分】潜空间采样与插值")
print("-" * 80)
with torch.no_grad():
    z1 = torch.randn(1, 8, device=device)
    z2 = torch.randn(1, 8, device=device)
    alphas = torch.linspace(0, 1, steps=5, device=device)
    for idx, alpha in enumerate(alphas):
        z_mix = (1 - alpha) * z1 + alpha * z2
        sample = model.decode(z_mix)
        print(f"插值 {idx + 1}: alpha={alpha.item():.2f}, 生成图均值={sample.mean().item():.4f}, std={sample.std().item():.4f}")
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ VAE 不是直接记住样本，而是学习一个连续潜空间
  ✓ 重参数化技巧让采样过程可微
  ✓ KL 散度约束潜变量分布接近标准高斯
  ✓ 训练完成后可以直接从高斯噪声采样生成新样本
"""
)
print("=" * 80)