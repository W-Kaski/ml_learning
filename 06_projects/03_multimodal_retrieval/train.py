#!/usr/bin/env python3
"""
03_multimodal_retrieval/train.py

对比学习训练流程：
  1. 加载合成图文对数据集
  2. 实例化双编码器
  3. Adam 优化器 + InfoNCE 对比捐失
  4. 每 epoch 在验证集计算对比损失，保存最优 checkpoint
  5. 导出训练曲线 JSON
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import build_dataloaders
from model import DualEncoder, info_nce_loss


# ============================================================
# 第一节：辅助函数
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for imgs, texts, _ in loader:
            imgs, texts = imgs.to(device), texts.to(device)
            sim, _, _ = model(imgs, texts)
            loss = info_nce_loss(sim)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ============================================================
# 第二节：主训练循环
# ============================================================

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 03_multimodal_retrieval 训练 ===")
    print(f"device={device}, epochs={config.epochs}, embed_dim={config.embed_dim}")

    train_loader, val_loader, _, token2idx = build_dataloaders(config)
    actual_vocab = len(token2idx)
    print(f"词表大小: {actual_vocab}  "
          f"训练 batch 数: {len(train_loader)}  验证 batch 数: {len(val_loader)}")

    model = DualEncoder(
        image_channels=config.image_channels,
        image_feature_dim=config.image_feature_dim,
        text_vocab_size=actual_vocab,
        embed_dim=config.embed_dim,
        temperature=config.temperature,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, texts, _ in train_loader:
            imgs, texts = imgs.to(device), texts.to(device)
            optimizer.zero_grad()
            sim, _, _ = model(imgs, texts)
            loss = info_nce_loss(sim)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss /= max(n_batches, 1)
        val_loss = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "logit_scale": round(model.logit_scale.exp().item(), 4),
        })
        print(f"Epoch {epoch:02d}/{config.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"scale={model.logit_scale.exp().item():.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.best_ckpt_path)
            print(f"  -> 保存 best checkpoint (val_loss={val_loss:.4f})")

    torch.save(model.state_dict(), config.last_ckpt_path)

    # 保存词表
    vocab_path = config.output_path / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(token2idx, f, ensure_ascii=False, indent=2)

    history_path = config.output_path / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n训练完成！最优验证损失: {best_val_loss:.4f}")
    print(f"输出文件:\n  {config.best_ckpt_path}\n  {config.last_ckpt_path}")
    print(f"  {history_path}\n  {vocab_path}")


if __name__ == "__main__":
    main()
