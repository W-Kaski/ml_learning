#!/usr/bin/env python3
"""
GPT 风格文本生成 - Tiny Causal Transformer
==========================================

学习内容：
- 因果掩码 (causal mask)
- 自回归训练目标 next-token prediction
- 简单字符级生成
- 温度采样
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


print("=" * 80)
print("【GPT 风格文本生成】Tiny Causal Transformer")
print("=" * 80)
print()

torch.manual_seed(42)

corpus = (
    "deep learning is fun. "
    "transformers learn context. "
    "gpt predicts the next token. "
    "attention lets tokens talk to each other. "
) * 8

chars = sorted(set(corpus))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in corpus], dtype=torch.long)

block_size = 24
batch_size = 16


def get_batch():
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out), weights


class Block(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        attn_out, weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, weights


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=48, num_heads=4, num_layers=2, block_size=24):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, block_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos).unsqueeze(0)
        last_weights = None
        for block in self.blocks:
            x, last_weights = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, last_weights

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyGPT(vocab_size=len(chars), block_size=block_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-3)

print("【第一部分】数据与词表")
print("-" * 80)
print(f"字符表大小: {len(chars)}")
print(f"训练语料长度: {len(data)}")
print(f"block_size: {block_size}, batch_size: {batch_size}")
print()

print("【第二部分】训练")
print("-" * 80)
for step in range(1, 121):
    x, y = get_batch()
    x = x.to(device)
    y = y.to(device)
    _, loss, weights = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if step % 30 == 0:
        print(f"step {step:3d}: loss={loss.item():.4f}, attn_mean={weights.mean().item():.4f}")
print()

print("【第三部分】生成示例")
print("-" * 80)
prompt = "transform"
context = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=60, temperature=0.9)[0].tolist()
text = "".join(itos[idx] for idx in generated)
print(f"prompt: {prompt}")
print(f"generated: {text}")
print()

print("【第四部分】因果掩码解释")
print("-" * 80)
mask = model.blocks[0].attn.mask[0, 0, :8, :8].cpu().int()
for row in mask:
    print("  ", row.tolist())
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ GPT 是 Decoder-only Transformer
  ✓ 训练目标是 next-token prediction
  ✓ 因果掩码保证当前位置不能偷看未来 token
  ✓ 推理时通过自回归逐个生成新 token

后续可以扩展：
  → Byte Pair Encoding 替代字符级建模
  → 更长上下文窗口与更多层数
  → top-k / top-p 采样
"""
)
print("=" * 80)