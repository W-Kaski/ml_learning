#!/usr/bin/env python3
"""
CIFAR-10 彩色图像分类 - ResNet18 教学版
=====================================

特点：
1. 详细中文注释，风格对齐 01_basics/tensor_basics.py
2. 默认 quick 模式，保证脚本可快速验证
3. 支持真实 CIFAR-10（可下载）与 FakeData（兜底）

运行示例：
  python3 02_cnn/cifar10_resnet.py --quick
  python3 02_cnn/cifar10_resnet.py --epochs 5 --batch-size 128
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


print("=" * 70)
print("🚀 CIFAR-10 ResNet18 训练开始")
print("=" * 70)


def build_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet18 教学脚本")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--quick", action="store_true", help="快速验证模式（使用子集数据）")
    return parser.parse_args()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders(batch_size, quick=False):
    """构建 CIFAR-10 数据加载器。

    quick=True 时，只取小子集，方便快速验证脚本是否能跑通。
    """
    print("\n【1. 数据准备】")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_root = "../datasets/CIFAR10"
    os.makedirs(data_root, exist_ok=True)

    try:
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
        print("✓ 使用真实 CIFAR-10 数据集")
    except Exception as err:
        print(f"⚠ CIFAR-10 下载/读取失败，切换到 FakeData 兜底: {err}")
        train_set = datasets.FakeData(size=4000, image_size=(3, 32, 32), num_classes=10, transform=train_tf)
        test_set = datasets.FakeData(size=800, image_size=(3, 32, 32), num_classes=10, transform=test_tf)

    if quick:
        # quick 模式：仅拿很小子集，方便 10~20 秒内完成验证
        train_set = Subset(train_set, range(min(1024, len(train_set))))
        test_set = Subset(test_set, range(min(256, len(test_set))))
        print("✓ quick 模式: 使用小子集进行快速训练验证")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练样本数: {len(train_set)}")
    print(f"测试样本数: {len(test_set)}")
    return train_loader, test_loader


def build_model(device):
    print("\n【2. 模型构建】")
    model = models.resnet18(weights=None)

    # ResNet18 默认输出 1000 类（ImageNet），这里改成 10 类（CIFAR-10）
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"✓ ResNet18 已创建，参数总数: {params:,}")
    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx:03d}/{len(loader):03d} | Loss {loss.item():.4f}")

    avg_loss = total_loss / total_count
    avg_acc = 100.0 * total_correct / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, 100.0 * total_correct / total_count


def main():
    args = build_args()
    device = get_device()

    print(f"设备: {device}")
    print(f"配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, quick={args.quick}")

    torch.manual_seed(42)

    train_loader, test_loader = get_data_loaders(args.batch_size, quick=args.quick)
    model = build_model(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\n【3. 开始训练】")
    best_acc = 0.0
    start_time = datetime.now()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print("-" * 70)
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/cifar10_resnet18_best.pth")
            print(f"✓ 保存最佳模型: ../models/cifar10_resnet18_best.pth (Acc={best_acc:.2f}%)")

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("✅ CIFAR-10 ResNet18 训练完成")
    print("=" * 70)
    print(f"总耗时: {elapsed}")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
