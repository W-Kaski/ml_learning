#!/usr/bin/env python3
"""
CNN 实战 - 数据增强（Augmentation）教学脚本
========================================

学习目标：
1. 理解为什么要做数据增强
2. 掌握常见图像增强操作
3. 对比“增强前后”的统计差异

说明：
- 本脚本默认使用 torchvision.FakeData，确保任何环境都可直接运行。
- 你后续可以把 FakeData 替换成 CIFAR-10 / 自定义数据集。
"""

import torch
from torchvision import datasets, transforms

print("=" * 70)
print("🧪 CNN 数据增强实验（Augmentation）")
print("=" * 70)


# ============================================================================
# 1. 定义两套 transform（基线 / 增强）
# ============================================================================
print("\n【1. 定义变换策略】")

# 基线：只做 ToTensor + Normalize（不做随机增强）
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 增强版：常见增强组合
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

print("基线策略: ToTensor + Normalize")
print("增强策略: Flip + Rotation + ColorJitter + RandomResizedCrop + Normalize")


# ============================================================================
# 2. 准备数据集（用 FakeData 方便快速验证）
# ============================================================================
print("\n【2. 构造数据集】")

base_dataset = datasets.FakeData(
    size=64,
    image_size=(3, 32, 32),
    num_classes=10,
    transform=base_transform,
)

aug_dataset = datasets.FakeData(
    size=64,
    image_size=(3, 32, 32),
    num_classes=10,
    transform=aug_transform,
)

print(f"基线数据集大小: {len(base_dataset)}")
print(f"增强数据集大小: {len(aug_dataset)}")


# ============================================================================
# 3. 对比同一样本在不同 transform 下的统计特征
# ============================================================================
print("\n【3. 增强效果对比】")

sample_idx = 0
base_img, base_label = base_dataset[sample_idx]
aug_img_1, aug_label_1 = aug_dataset[sample_idx]
aug_img_2, aug_label_2 = aug_dataset[sample_idx]  # 同一个索引再取一次，会有随机变化

print(f"样本标签（基线/增强）: {base_label} / {aug_label_1} / {aug_label_2}")
print(f"图像形状: {tuple(base_img.shape)}")

print("\n--- 像素统计（mean/std/min/max）---")
print(f"基线   : mean={base_img.mean():.4f}, std={base_img.std():.4f}, min={base_img.min():.4f}, max={base_img.max():.4f}")
print(f"增强#1 : mean={aug_img_1.mean():.4f}, std={aug_img_1.std():.4f}, min={aug_img_1.min():.4f}, max={aug_img_1.max():.4f}")
print(f"增强#2 : mean={aug_img_2.mean():.4f}, std={aug_img_2.std():.4f}, min={aug_img_2.min():.4f}, max={aug_img_2.max():.4f}")

# 计算同一索引两次增强图的差异（验证随机性）
diff = (aug_img_1 - aug_img_2).abs().mean().item()
print(f"两次随机增强图像的平均绝对差异: {diff:.4f}")


# ============================================================================
# 4. 小批量检查（模拟训练输入）
# ============================================================================
print("\n【4. 批量数据检查】")

loader = torch.utils.data.DataLoader(aug_dataset, batch_size=8, shuffle=True)
images, labels = next(iter(loader))

print(f"batch 图像形状: {tuple(images.shape)}")
print(f"batch 标签形状: {tuple(labels.shape)}")
print(f"标签示例: {labels.tolist()}")


# ============================================================================
# 5. 实战建议
# ============================================================================
print("\n【5. 实战建议】")
print("1) 小数据集：增强强度可以稍大，缓解过拟合")
print("2) 大数据集：增强适中，避免破坏样本语义")
print("3) 验证/测试集：通常不做随机增强，保持评估一致性")
print("4) 先做可解释增强（翻转/旋转），再逐步加入复杂增强")


print("\n" + "=" * 70)
print("✅ augmentation.py 运行完成")
print("=" * 70)
