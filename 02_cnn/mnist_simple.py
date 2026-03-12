#!/usr/bin/env python3
"""
MNIST 手写数字识别 - 简化版 CNN
==================================

使用纯 PyTorch 实现，不依赖 torchvision
适合快速开始学习！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gzip
import numpy as np
import urllib.request
import os
from datetime import datetime

# ============================================================================
# 配置
# ============================================================================
class Config:
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "../datasets/MNIST_raw"
    model_dir = "../models"
    seed = 42

# ============================================================================
# 手动下载 MNIST 数据集
# ============================================================================
def download_mnist(data_dir):
    """下载原始 MNIST 数据"""
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"正在下载 {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    print("✓ 数据集下载完成")

def load_mnist_images(filename):
    """加载 MNIST 图像"""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    """加载 MNIST 标签"""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

class MNISTDataset(Dataset):
    """MNIST 数据集类"""
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).unsqueeze(1)  # 添加通道维度
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ============================================================================
# CNN 模型
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ============================================================================
# 训练与测试
# ============================================================================
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'\nEpoch {epoch} - 训练集: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
    
    return avg_loss, accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'测试集: Loss={test_loss:.4f}, Accuracy={correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# ============================================================================
# 主函数
# ============================================================================
def main():
    config = Config()
    
    print("=" * 70)
    print("🚀 MNIST CNN 训练 (简化版)")
    print("=" * 70)
    print(f"设备: {config.device}")
    print(f"批大小: {config.batch_size}, Epochs: {config.epochs}, 学习率: {config.learning_rate}")
    print()
    
    torch.manual_seed(config.seed)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # 下载并加载数据
    try:
        download_mnist(config.data_dir)
        
        train_images = load_mnist_images(f'{config.data_dir}/train-images-idx3-ubyte.gz')
        train_labels = load_mnist_labels(f'{config.data_dir}/train-labels-idx1-ubyte.gz')
        test_images = load_mnist_images(f'{config.data_dir}/t10k-images-idx3-ubyte.gz')
        test_labels = load_mnist_labels(f'{config.data_dir}/t10k-labels-idx1-ubyte.gz')
        
        print(f"训练集: {len(train_labels)} 张图片")
        print(f"测试集: {len(test_labels)} 张图片\n")
        
        train_dataset = MNISTDataset(train_images, train_labels)
        test_dataset = MNISTDataset(test_images, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True)
    
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("提示: 可能是网络问题，请稍后重试或检查防火墙设置")
        return
    
    # 创建模型
    model = SimpleCNN().to(config.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建完成 ({params:,} 参数)\n")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练
    best_acc = 0
    start_time = datetime.now()
    
    for epoch in range(1, config.epochs + 1):
        print(f"{'='*70}")
        train_loss, train_acc = train_epoch(model, config.device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, config.device, test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, f'{config.model_dir}/mnist_cnn_simple.pth')
            print(f"✓ 最佳模型已保存 (准确率: {test_acc:.2f}%)\n")
    
    # 总结
    end_time = datetime.now()
    print("=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    print(f"总耗时: {end_time - start_time}")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print("\n📚 下一步:")
    print("  1. 查看训练好的模型: models/mnist_cnn_simple.pth")
    print("  2. 尝试修改网络架构（添加/删除层）")
    print("  3. 调整超参数并对比结果")
    print("  4. 学习路线图: cat README.md")
    print("=" * 70)

if __name__ == '__main__':
    main()
