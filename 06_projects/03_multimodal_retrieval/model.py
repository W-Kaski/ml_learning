#!/usr/bin/env python3
"""
03_multimodal_retrieval/model.py

双编码器（Dual Encoder）跨模态检索模型。

架构：
  图像编码器 ─ CNN (Conv×3 → pool → flatten) → Linear → L2 normalize → embed_dim
  文本编码器 ─ Embedding → 均值池化 → Linear → L2 normalize → embed_dim

训练目标：InfoNCE 对比损失（与 CLIP 相同思路）
  - 正样本：(image_i, text_i) 来自同一类别
  - 负样本：同 batch 内其他不同类别的图文对
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一节：图像编码器
# ============================================================

class ImageEncoder(nn.Module):
    """轻量 CNN：3 层 Conv + BN + ReLU + MaxPool → 全连接投影。"""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, embed_dim), L2 归一化"""
        feat = self.cnn(x).squeeze(-1).squeeze(-1)
        emb = self.proj(feat)
        return F.normalize(emb, p=2, dim=-1)


# ============================================================
# 第二节：文本编码器
# ============================================================

class TextEncoder(nn.Module):
    """Embedding + 均值池化（忽略 PAD）→ 全连接投影。"""

    def __init__(self, vocab_size: int = 200, embed_dim: int = 64, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) token ids -> (B, embed_dim), L2 归一化"""
        emb = self.embedding(x)
        mask = (x != self.pad_idx).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        emb_out = self.proj(pooled)
        return F.normalize(emb_out, p=2, dim=-1)


# ============================================================
# 第三节：双编码器 + InfoNCE 损失
# ============================================================

class DualEncoder(nn.Module):
    """将 ImageEncoder 和 TextEncoder 组合，共享可学习温度参数 logit_scale。"""

    def __init__(
        self,
        image_channels: int = 3,
        image_feature_dim: int = 128,
        text_vocab_size: int = 200,
        embed_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.image_enc = ImageEncoder(image_channels, image_feature_dim, embed_dim)
        self.text_enc  = TextEncoder(text_vocab_size, embed_dim)
        init_val = -torch.tensor(temperature).log()
        self.logit_scale = nn.Parameter(init_val.clone().detach())

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_enc(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_enc(text)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        """返回相似度矩阵 (B, B) 及双侧嵌入。"""
        img_emb  = self.encode_image(image)
        text_emb = self.encode_text(text)
        scale = self.logit_scale.exp().clamp(max=100)
        sim = scale * (img_emb @ text_emb.T)
        return sim, img_emb, text_emb


# ============================================================
# 第四节：InfoNCE 损失
# ============================================================

def info_nce_loss(sim: torch.Tensor) -> torch.Tensor:
    """对称 InfoNCE：对角线为正样本，其余为负样本。"""
    B = sim.size(0)
    labels = torch.arange(B, device=sim.device)
    loss_i2t = F.cross_entropy(sim,   labels)
    loss_t2i = F.cross_entropy(sim.T, labels)
    return (loss_i2t + loss_t2i) / 2


if __name__ == "__main__":
    model = DualEncoder()
    imgs  = torch.randn(4, 3, 32, 32)
    texts = torch.randint(0, 50, (4, 16))
    sim, img_emb, text_emb = model(imgs, texts)
    loss = info_nce_loss(sim)
    print(f"图像嵌入: {img_emb.shape}  L2 norm={img_emb.norm(dim=-1).mean():.4f}")
    print(f"文本嵌入: {text_emb.shape}")
    print(f"相似度矩阵: {sim.shape}")
    print(f"InfoNCE loss: {loss.item():.4f}")
