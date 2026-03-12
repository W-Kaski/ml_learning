#!/usr/bin/env python3
"""
DCGAN 图像生成 - 对抗训练教学版
=================================

学习内容：
- 生成器与判别器的博弈
- 卷积转置上采样
- BCE 对抗损失
- 假样本质量与判别器分数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


print("=" * 80)
print("【DCGAN 图像生成】对抗训练入门")
print("=" * 80)
print()

torch.manual_seed(42)


class Generator(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = datasets.FakeData(size=256, image_size=(3, 32, 32), num_classes=10, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

print("【第一部分】网络结构")
print("-" * 80)
print(f"Generator 参数量: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator 参数量: {sum(p.numel() for p in discriminator.parameters()):,}")
print()

print("【第二部分】对抗训练")
print("-" * 80)
latent_dim = 32
for epoch in range(1, 4):
    d_loss_value = 0.0
    g_loss_value = 0.0
    real_score_value = 0.0
    fake_score_value = 0.0
    steps = 0
    for real_images, _ in loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)

        d_optimizer.zero_grad()
        real_scores = discriminator(real_images)
        fake_scores = discriminator(fake_images.detach())
        d_loss_real = criterion(real_scores, real_labels)
        d_loss_fake = criterion(fake_scores, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        fake_scores_for_g = discriminator(fake_images)
        g_loss = criterion(fake_scores_for_g, real_labels)
        g_loss.backward()
        g_optimizer.step()

        d_loss_value += d_loss.item()
        g_loss_value += g_loss.item()
        real_score_value += real_scores.mean().item()
        fake_score_value += fake_scores.mean().item()
        steps += 1

    print(
        f"Epoch {epoch}: d_loss={d_loss_value / steps:.4f}, g_loss={g_loss_value / steps:.4f}, "
        f"D(real)={real_score_value / steps:.4f}, D(fake)={fake_score_value / steps:.4f}"
    )
print()

print("【第三部分】生成样本统计")
print("-" * 80)
with torch.no_grad():
    noise = torch.randn(4, latent_dim, 1, 1, device=device)
    fake_images = generator(noise)
    scores = discriminator(fake_images)
    for idx in range(4):
        print(
            f"样本 {idx + 1}: mean={fake_images[idx].mean().item():.4f}, "
            f"std={fake_images[idx].std().item():.4f}, D(x)={scores[idx].item():.4f}"
        )
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ GAN 训练是生成器与判别器的双人博弈
  ✓ 生成器目标是骗过判别器，判别器目标是区分真伪
  ✓ DCGAN 用卷积和转置卷积稳定图像生成训练
  ✓ 对抗训练常见问题包括模式崩塌和训练不稳定
"""
)
print("=" * 80)