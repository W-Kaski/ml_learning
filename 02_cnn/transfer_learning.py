#!/usr/bin/env python3
"""
CNN 实战 - 迁移学习（Transfer Learning）教学脚本
===============================================

学习目标：
1. 理解“冻结特征提取层 + 替换分类头”的迁移学习流程
2. 学会统计可训练参数量
3. 完成一次可运行的最小训练验证

说明：
- 默认使用 FakeData，保证脚本快速可跑。
- 你后续可把 FakeData 换成 CIFAR-10 或自定义数据集。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


print("=" * 70)
print("🔁 CNN 迁移学习实验（Transfer Learning）")
print("=" * 70)


# ============================================================================
# 1. 配置
# ============================================================================
print("\n【1. 实验配置】")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
batch_size = 32
epochs = 1  # 教学脚本先跑 1 轮做冒烟验证

print(f"设备: {device}")
print(f"类别数: {num_classes}, batch_size: {batch_size}, epochs: {epochs}")


# ============================================================================
# 2. 数据准备（FakeData）
# ============================================================================
print("\n【2. 数据准备】")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 输入推荐尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_set = datasets.FakeData(
    size=256,
    image_size=(3, 224, 224),
    num_classes=num_classes,
    transform=transform,
)
val_set = datasets.FakeData(
    size=64,
    image_size=(3, 224, 224),
    num_classes=num_classes,
    transform=transform,
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"训练样本: {len(train_set)}, 验证样本: {len(val_set)}")


# ============================================================================
# 3. 模型构建：ResNet18 迁移学习
# ============================================================================
print("\n【3. 模型构建】")

# 为了离线可跑，这里不下载预训练权重；教学流程与真实迁移学习一致
model = models.resnet18(weights=None)

# 3.1 冻结骨干网络参数
for param in model.parameters():
    param.requires_grad = False

# 3.2 替换最后分类层（fc）
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# 仅分类头可训练
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

model = model.to(device)

print(f"模型总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}（仅 fc 层）")


# ============================================================================
# 4. 训练组件
# ============================================================================
print("\n【4. 训练组件】")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)  # 只优化分类头

print("损失函数: CrossEntropyLoss")
print("优化器: Adam(model.fc.parameters())")


# ============================================================================
# 5. 训练 + 验证
# ============================================================================
print("\n【5. 开始训练】")

model.train()
running_loss = 0.0
running_correct = 0
running_total = 0

for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    preds = logits.argmax(dim=1)
    running_correct += (preds == labels).sum().item()
    running_total += images.size(0)

    if batch_idx % 5 == 0:
        print(f"Batch {batch_idx:02d}/{len(train_loader):02d} | Loss={loss.item():.4f}")

train_loss = running_loss / running_total
train_acc = 100.0 * running_correct / running_total

# 验证
model.eval()
val_loss_sum = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        val_loss_sum += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += images.size(0)

val_loss = val_loss_sum / val_total
val_acc = 100.0 * val_correct / val_total


# ============================================================================
# 6. 结果总结
# ============================================================================
print("\n【6. 结果总结】")
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

print("\n" + "=" * 70)
print("✅ transfer_learning.py 运行完成")
print("=" * 70)
print("\n💡 关键要点：")
print("  1) 冻结骨干网络，减少训练参数和训练时间")
print("  2) 仅训练分类头，适合小数据集快速迁移")
print("  3) 后续可分阶段解冻（先训练 fc，再微调 backbone）")
print("=" * 70)
