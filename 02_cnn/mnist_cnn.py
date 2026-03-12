#!/usr/bin/env python3
"""
MNIST 手写数字识别 - CNN 实现
=================================

学习目标：
1. 理解 CNN 基本组件（卷积、池化、全连接）
2. 掌握完整的训练流程
3. 学会模型评估和可视化

硬件：RTX 4080 (16GB)
预期准确率：>99%
训练时间：~2分钟
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# ============================================================================
# 1. 超参数配置
# ============================================================================

class Config:
    """超参数集中管理"""
    # 训练参数
    batch_size = 64          # 批大小
    test_batch_size = 1000   # 测试批大小
    epochs = 10              # 训练轮数
    learning_rate = 0.001    # 学习率
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据路径
    data_dir = "../datasets/MNIST"
    model_dir = "../models"
    
    # 随机种子（确保可复现）
    seed = 42
    
    def __str__(self):
        return f"Config(lr={self.learning_rate}, batch={self.batch_size}, epochs={self.epochs})"


# ============================================================================
# 2. 数据准备
# ============================================================================

def get_data_loaders(config):
    """
    准备 MNIST 数据集
    
    数据增强策略：
    - 训练集：随机旋转、归一化
    - 测试集：仅归一化
    """
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # 随机旋转 ±10度
        transforms.ToTensor(),              # 转为张量 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 统计均值/标准差
    ])
    
    # 测试数据（不增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载并加载数据集
    train_dataset = datasets.MNIST(
        config.data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        config.data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,          # 训练时打乱
        num_workers=2,         # 多线程加载
        pin_memory=True        # GPU 加速
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ 数据加载完成")
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  测试集: {len(test_dataset)} 张图片")
    
    return train_loader, test_loader


# ============================================================================
# 3. CNN 模型定义
# ============================================================================

class SimpleCNN(nn.Module):
    """
    经典 LeNet-5 风格 CNN
    
    架构：
    Input(28x28x1) → Conv1(24x24x32) → Pool(12x12x32) 
                   → Conv2(8x8x64)   → Pool(4x4x64)
                   → FC(128) → FC(10)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)   # 1→32 通道, 5x5 卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # 32→64 通道
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化
        
        # Dropout 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 展平后连接
        self.fc2 = nn.Linear(128, 10)          # 输出 10 类
    
    def forward(self, x):
        """前向传播"""
        # 卷积块 1
        x = self.conv1(x)         # [B, 1, 28, 28] → [B, 32, 24, 24]
        x = F.relu(x)             # 激活
        x = self.pool(x)          # [B, 32, 12, 12]
        
        # 卷积块 2
        x = self.conv2(x)         # [B, 32, 12, 12] → [B, 64, 8, 8]
        x = F.relu(x)
        x = self.pool(x)          # [B, 64, 4, 4]
        x = self.dropout1(x)
        
        # 展平
        x = x.view(-1, 64 * 4 * 4)  # [B, 1024]
        
        # 全连接层
        x = self.fc1(x)           # [B, 128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)           # [B, 10]
        
        return F.log_softmax(x, dim=1)  # 对数概率


# ============================================================================
# 4. 训练与评估函数
# ============================================================================

def train_epoch(model, device, train_loader, optimizer, epoch):
    """训练一个 epoch"""
    model.train()  # 训练模式（启用 Dropout）
    
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移到 GPU
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = F.nll_loss(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 打印进度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    return avg_loss, accuracy


def test(model, device, test_loader):
    """测试模型"""
    model.eval()  # 评估模式（禁用 Dropout）
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 累计损失
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # 统计正确数
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\n测试集: 平均损失: {test_loss:.4f}, '
          f'准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


# ============================================================================
# 5. 可视化函数
# ============================================================================

def visualize_predictions(model, device, test_loader, num_images=10):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 预测
    with torch.no_grad():
        images_gpu = images.to(device)
        output = model(images_gpu)
        predictions = output.argmax(dim=1).cpu().numpy()
    
    # 绘图
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        
        # 显示图像
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        
        # 标题：真实标签 vs 预测
        true_label = labels[i].item()
        pred_label = predictions[i]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'真: {true_label}, 预测: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../models/mnist_predictions.png', dpi=150)
    print("✓ 可视化结果已保存到 models/mnist_predictions.png")
    plt.close()


def plot_training_history(history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', linewidth=2, label='Test Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curve', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加整体标题
    fig.suptitle('MNIST CNN Training History', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png', dpi=150, bbox_inches='tight')
    print("✓ 训练曲线已保存到 models/training_history.png")
    plt.close()


# ============================================================================
# 6. 主函数
# ============================================================================

def main():
    # 配置
    config = Config()
    print("=" * 70)
    print("🚀 MNIST CNN 训练开始")
    print("=" * 70)
    print(f"设备: {config.device}")
    print(config)
    print()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    
    # 创建目录
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = get_data_loaders(config)
    print()
    
    # 创建模型
    model = SimpleCNN().to(config.device)
    print(f"✓ 模型已创建并移至 {config.device}")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 训练循环
    best_acc = 0
    start_time = datetime.now()
    
    for epoch in range(1, config.epochs + 1):
        print(f"{'='*70}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, config.device, train_loader, optimizer, epoch)
        
        # 测试
        test_loss, test_acc = test(model, config.device, test_loader)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, f'{config.model_dir}/mnist_cnn_best.pth')
            print(f"✓ 最佳模型已保存 (准确率: {test_acc:.2f}%)")
    
    # 训练总结
    end_time = datetime.now()
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    print(f"总耗时: {end_time - start_time}")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"最终训练准确率: {history['train_acc'][-1]:.2f}%")
    print(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")
    print()
    
    # 可视化
    print("生成可视化结果...")
    visualize_predictions(model, config.device, test_loader)
    plot_training_history(history)
    
    print("\n" + "=" * 70)
    print("📚 下一步学习建议：")
    print("  1. 修改网络架构（添加更多卷积层）")
    print("  2. 调整超参数（学习率、batch size）")
    print("  3. 尝试不同优化器（SGD、RMSprop）")
    print("  4. 实现学习率调度器")
    print("  5. 进入 CIFAR-10 彩色图像分类")
    print("=" * 70)


if __name__ == '__main__':
    main()
