#!/usr/bin/env python3
"""
Vision Transformer (ViT) - 图像分类教学版
=========================================

学习内容：
- Patch Embedding
- [CLS] token 图像级表示
- Transformer Encoder 做视觉建模
- FakeData 上的快速训练验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


print("=" * 80)
print("【Vision Transformer】Patch 到 Token 的图像分类")
print("=" * 80)
print()

torch.manual_seed(42)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TinyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        b = x.size(0)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.encoder(x)
        cls_repr = self.norm(x[:, 0])
        return self.head(cls_repr), cls_repr


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = datasets.FakeData(size=256, image_size=(3, 32, 32), num_classes=10, transform=transform)
train_set, val_set = random_split(dataset, [192, 64], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("【第一部分】Patch Embedding")
print("-" * 80)
images, labels = next(iter(train_loader))
patches = model.patch_embed(images[:2].to(device))
print(f"原图形状: {images[:2].shape}")
print(f"patch token 形状: {patches.shape}")
print(f"每张图的 patch 数量: {patches.size(1)}")
print()

print("【第二部分】模型训练")
print("-" * 80)
for epoch in range(1, 4):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        logits, cls_repr = model(batch_images)
        loss = criterion(logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_images.size(0)
        total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        total_seen += batch_images.size(0)

    train_loss = total_loss / total_seen
    train_acc = total_correct / total_seen

    model.eval()
    val_correct = 0
    val_seen = 0
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            logits, _ = model(batch_images)
            val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            val_seen += batch_images.size(0)
    val_acc = val_correct / val_seen
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, val_acc={val_acc:.2%}, cls_norm={cls_repr.norm(dim=1).mean().item():.4f}")
print()

print("【第三部分】ViT 关键观察")
print("-" * 80)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print("ViT 核心流程: 图像 -> patch -> token -> [CLS] 聚合 -> 分类头")
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ ViT 先把图像切成 patch，再把 patch 当作序列 token
  ✓ [CLS] token 负责聚合整张图的全局表示
  ✓ 位置编码仍然必要，因为 patch 顺序也有空间含义
  ✓ 小数据集上 ViT 通常不如 CNN 稳定，大模型更依赖预训练
"""
)
print("=" * 80)