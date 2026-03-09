#!/usr/bin/env python3
"""
PyTorch 基础 - DataLoader 与数据增强
===================================

学习目标：
1. 理解 Dataset / DataLoader 的作用
2. 学会按 batch 读取数据
3. 体验 transform 数据增强流程
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("=" * 70)
print("🔥 PyTorch 数据加载（DataLoader）")
print("=" * 70)

# ============================================================================
# 1. 定义数据增强
# ============================================================================
print("\n【1. 定义 transform】")

# 这里用 FakeData，避免下载数据，任何机器都能快速跑
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

print("训练增强: 随机翻转 + 随机旋转 + ToTensor")
print("测试增强: 仅 ToTensor")

# ============================================================================
# 2. 构造 Dataset
# ============================================================================
print("\n【2. 构造 Dataset】")

train_dataset = datasets.FakeData(
    size=128,
    image_size=(3, 32, 32),
    num_classes=10,
    transform=train_transform,
)

test_dataset = datasets.FakeData(
    size=32,
    image_size=(3, 32, 32),
    num_classes=10,
    transform=test_transform,
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# ============================================================================
# 3. 构造 DataLoader
# ============================================================================
print("\n【3. 构造 DataLoader】")

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

print("train_loader: batch_size=16, shuffle=True")
print("test_loader:  batch_size=8, shuffle=False")

# ============================================================================
# 4. 读取一个 batch 看看
# ============================================================================
print("\n【4. 读取 batch】")

train_images, train_labels = next(iter(train_loader))
test_images, test_labels = next(iter(test_loader))

print(f"训练 batch 图像形状: {tuple(train_images.shape)}")
print(f"训练 batch 标签形状: {tuple(train_labels.shape)}")
print(f"测试 batch 图像形状: {tuple(test_images.shape)}")
print(f"测试 batch 标签形状: {tuple(test_labels.shape)}")

print(f"像素值范围（训练 batch）: [{train_images.min().item():.3f}, {train_images.max().item():.3f}]")
print(f"前 8 个训练标签: {train_labels[:8].tolist()}")

# ============================================================================
# 5. 遍历 DataLoader（模拟训练循环）
# ============================================================================
print("\n【5. 遍历 DataLoader】")

total_samples = 0
for batch_idx, (images, labels) in enumerate(train_loader):
    total_samples += images.size(0)
    if batch_idx < 2:
        print(f"Batch {batch_idx}: images={tuple(images.shape)}, labels={tuple(labels.shape)}")

print(f"总共遍历样本数: {total_samples}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ DataLoader 基础完成！")
print("=" * 70)
print("\n💡 关键要点：")
print("  1. Dataset 负责定义数据，DataLoader 负责批量加载")
print("  2. 训练集常用 shuffle=True，测试集一般 False")
print("  3. transform 可以放在 Dataset 中，自动对每个样本生效")
print("=" * 70)
