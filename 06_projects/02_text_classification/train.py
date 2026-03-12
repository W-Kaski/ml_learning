#!/usr/bin/env python3
"""
02_text_classification/train.py

完整训练流程：
  1. 加载合成情感数据集，构建词表
  2. 实例化双向 GRU 分类器
  3. Adam 优化器 + 交叉熵损失
  4. 训练循环：每 epoch 结束后在验证集上评估
  5. 保存 best / last checkpoint 及训练曲线 JSON
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

# 允许从当前目录直接 import
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_processor import build_dataloaders
from model import SimpleTextClassifier


# ============================================================
# 第一节：辅助函数
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# ============================================================
# 第二节：主训练循环
# ============================================================

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 02_text_classification 训练 ===")
    print(f"device={device}, epochs={config.epochs}, batch_size={config.batch_size}")

    # --- 数据 ---
    train_loader, val_loader, _, vocab = build_dataloaders(config)
    actual_vocab_size = len(vocab)
    print(f"词表大小: {actual_vocab_size}  "
          f"训练集: {config.train_size}  验证集: {config.val_size}")

    # --- 模型（使用实际词表大小，避免 OOB 错误）---
    model = SimpleTextClassifier(
        vocab_size=actual_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = []
    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        # --- 训练 ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        print(f"Epoch {epoch:02d}/{config.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # 保存最优 checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.best_ckpt_path)
            print(f"  -> 保存 best checkpoint (val_acc={val_acc:.4f})")

    # 保存最后 checkpoint
    torch.save(model.state_dict(), config.last_ckpt_path)

    # 导出训练曲线
    history_path = config.output_path / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 保存词表（供 infer.py 使用）
    vocab_path = config.output_path / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab.token2idx, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print(f"输出文件:\n  {config.best_ckpt_path}\n  {config.last_ckpt_path}")
    print(f"  {history_path}\n  {vocab_path}")


if __name__ == "__main__":
    main()
