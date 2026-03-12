#!/usr/bin/env python3
"""
Transformer 从零实现 - 核心模块拆解
=================================

学习内容：
- 缩放点积注意力
- 多头注意力
- 位置编码
- 前馈网络与残差连接
- Transformer Encoder Block 前向传播
"""

import math

import torch
import torch.nn as nn


print("=" * 80)
print("【Transformer 从零实现】核心结构教学版")
print("=" * 80)
print()

torch.manual_seed(42)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context), attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


print("【第一部分】输入嵌入与位置编码")
print("-" * 80)

batch_size = 2
seq_len = 6
vocab_size = 20
d_model = 16
num_heads = 4

token_ids = torch.tensor(
    [
        [1, 5, 7, 9, 2, 0],
        [3, 4, 8, 6, 0, 0],
    ]
)
embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
pos_encoding = PositionalEncoding(d_model, max_len=32)

embedded = embedding(token_ids)
encoded = pos_encoding(embedded)

print(f"token_ids 形状: {token_ids.shape}")
print(f"embedding 输出: {embedded.shape}")
print(f"加入位置编码后: {encoded.shape}")
print(f"第一个 token 的前 6 维: {encoded[0, 0, :6]}")
print()

print("【第二部分】缩放点积注意力")
print("-" * 80)

mask = (token_ids != 0).unsqueeze(1).unsqueeze(2)
attention = MultiHeadSelfAttention(d_model, num_heads)
attn_out, attn_weights = attention(encoded, mask=mask)

print(f"注意力输出形状: {attn_out.shape}")
print(f"注意力权重形状: {attn_weights.shape} = (B, heads, T, T)")
print("第 1 个样本第 1 个头的注意力分布:")
for t in range(seq_len):
    row = attn_weights[0, 0, t]
    print(f"  query t={t}: {row.tolist()}")
print()

print("【第三部分】Encoder Block 前向传播")
print("-" * 80)

block = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, mlp_dim=32, dropout=0.0)
block_out, block_weights = block(encoded, mask=mask)

print(f"Block 输出形状: {block_out.shape}")
print(f"输出均值: {block_out.mean().item():.4f}")
print(f"输出标准差: {block_out.std().item():.4f}")
print()

print("【第四部分】堆叠两个 Encoder Block")
print("-" * 80)

stack = nn.ModuleList([
    TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, mlp_dim=32, dropout=0.0),
    TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, mlp_dim=32, dropout=0.0),
])

x = encoded
for idx, layer in enumerate(stack, start=1):
    x, weights = layer(x, mask=mask)
    print(f"  第 {idx} 层输出范数: {x.norm().item():.4f}")
print()

print("【第五部分】参数量拆解")
print("-" * 80)
for name, param in block.named_parameters():
    print(f"{name:24s}: {tuple(param.shape)} = {param.numel():,}")
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ Self-Attention 让每个 token 直接看到整段上下文
  ✓ Multi-Head 通过多个子空间并行建模不同关系
  ✓ Positional Encoding 弥补注意力本身缺少顺序信息的问题
  ✓ Encoder Block = 注意力 + 前馈网络 + 残差 + LayerNorm

下一步：
  → BERT：只用 Encoder，适合理解式任务
  → GPT：带因果掩码的 Decoder，适合生成式任务
  → ViT：把图像 patch 当作 token 输入 Transformer
"""
)
print("=" * 80)