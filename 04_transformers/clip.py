#!/usr/bin/env python3
"""
CLIP 教学版 - 图文对比学习
==========================

学习内容：
- 双塔编码器 (image encoder / text encoder)
- 共享嵌入空间
- 对比学习损失 InfoNCE
- 图文检索的基本思路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


print("=" * 80)
print("【CLIP 教学版】图文对比学习")
print("=" * 80)
print()

torch.manual_seed(42)

concepts = [
    "cat", "dog", "car", "tree", "house", "flower", "book", "phone"
]
text_templates = [f"a photo of {word}" for word in concepts]

vocab = {"[PAD]": 0, "[UNK]": 1}
for sentence in text_templates:
    for token in sentence.split():
        if token not in vocab:
            vocab[token] = len(vocab)


def encode_sentence(sentence, max_len=5):
    ids = [vocab.get(tok, vocab["[UNK]"]) for tok in sentence.split()][:max_len]
    if len(ids) < max_len:
        ids += [vocab["[PAD]"]] * (max_len - len(ids))
    return ids


text_ids = torch.tensor([encode_sentence(sentence) for sentence in text_templates], dtype=torch.long)

# 用概念 id 生成带结构的“伪图像特征”，保证脚本可跑且能学到对应关系。
image_features = []
for idx in range(len(concepts)):
    base = torch.randn(32) * 0.05
    base[idx % 8] += 1.5
    base[(idx + 3) % 16] += 1.0
    image_features.append(base)
image_features = torch.stack(image_features)


class ImageEncoder(nn.Module):
    def __init__(self, in_dim=32, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=64,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.proj = nn.Linear(16, embed_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x, src_key_padding_mask=input_ids == 0)
        sentence_repr = x.mean(dim=1)
        return F.normalize(self.proj(sentence_repr), dim=-1)


class TinyCLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image_x, text_x):
        image_emb = self.image_encoder(image_x)
        text_emb = self.text_encoder(text_x)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * image_emb @ text_emb.t()
        return logits, image_emb, text_emb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCLIP(vocab_size=len(vocab)).to(device)
image_features = image_features.to(device)
text_ids = text_ids.to(device)
targets = torch.arange(len(concepts), device=device)
optimizer = optim.AdamW(model.parameters(), lr=2e-3)

print("【第一部分】输入表示")
print("-" * 80)
print(f"文本张量: {text_ids.shape}")
print(f"图像特征张量: {image_features.shape}")
print(f"词表大小: {len(vocab)}")
print()

print("【第二部分】对比学习训练")
print("-" * 80)
for epoch in range(1, 121):
    logits, image_emb, text_emb = model(image_features, text_ids)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    loss = (loss_i + loss_t) / 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 30 == 0:
        retrieval_acc = (logits.argmax(dim=1) == targets).float().mean().item()
        print(f"epoch {epoch:3d}: loss={loss.item():.4f}, image->text acc={retrieval_acc:.2%}, scale={model.logit_scale.exp().item():.2f}")
print()

print("【第三部分】相似度矩阵")
print("-" * 80)
model.eval()
with torch.no_grad():
    logits, image_emb, text_emb = model(image_features, text_ids)
    probs = torch.softmax(logits, dim=1)

header = "       " + " ".join(f"{word:>8s}" for word in concepts[:6])
print(header)
for i, word in enumerate(concepts[:6]):
    row = " ".join(f"{probs[i, j].item():8.3f}" for j in range(6))
    print(f"{word:>6s} {row}")
print()

print("【第四部分】检索演示")
print("-" * 80)
for i, word in enumerate(concepts[:4]):
    best_idx = logits[i].argmax().item()
    print(f"图像 '{word}' 最匹配文本: '{text_templates[best_idx]}'")
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ CLIP 的目标是把图像和文本映射到同一个向量空间
  ✓ 正样本对相似度高，负样本对相似度低
  ✓ 训练后可以直接做 zero-shot 分类或跨模态检索
  ✓ 对比学习损失本质上是在做批内匹配
"""
)
print("=" * 80)