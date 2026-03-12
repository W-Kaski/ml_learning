#!/usr/bin/env python3
"""
BERT 风格文本分类 - 教学简化版
================================

学习内容：
- Token Embedding + Position Embedding
- Transformer Encoder 做句子建模
- [CLS] token 作为分类表示
- 文本分类训练循环

说明：
- 环境里未安装 transformers，本脚本使用 PyTorch 原生模块实现 BERT 风格结构。
- 重点是理解 BERT 的输入组织与 Encoder 分类流程。
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim


print("=" * 80)
print("【BERT 风格文本分类】Encoder-only 分类流程")
print("=" * 80)
print()

torch.manual_seed(42)


samples = [
    ("this movie is wonderful and touching", 1),
    ("absolutely boring film with weak acting", 0),
    ("great story and excellent cast", 1),
    ("the plot is dull and predictable", 0),
    ("smart script with strong emotional payoff", 1),
    ("waste of time and money", 0),
    ("visually stunning and deeply moving", 1),
    ("bad pacing and terrible dialogue", 0),
    ("heartwarming performance from the lead", 1),
    ("messy editing and uninspired direction", 0),
]

special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
counter = Counter()
for text, _ in samples:
    counter.update(text.split())

vocab = {token: idx for idx, token in enumerate(special_tokens)}
for word, _ in counter.most_common():
    if word not in vocab:
        vocab[word] = len(vocab)

id_to_token = {idx: token for token, idx in vocab.items()}
max_len = 10


def encode_text(text):
    tokens = ["[CLS]"] + text.split() + ["[SEP]"]
    ids = [vocab.get(tok, vocab["[UNK]"]) for tok in tokens][:max_len]
    attention_mask = [1] * len(ids)
    if len(ids) < max_len:
        pad_length = max_len - len(ids)
        ids += [vocab["[PAD]"]] * pad_length
        attention_mask += [0] * pad_length
    return ids, attention_mask


input_ids = []
attention_masks = []
labels = []
for text, label in samples:
    ids, mask = encode_text(text)
    input_ids.append(ids)
    attention_masks.append(mask)
    labels.append(label)

input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

print("【第一部分】输入组织")
print("-" * 80)
print(f"词表大小: {len(vocab)}")
print(f"输入张量: {input_ids.shape}")
print(f"注意力掩码: {attention_masks.shape}")
print(f"示例 token id: {input_ids[0].tolist()}")
print(f"对应 token: {[id_to_token[idx] for idx in input_ids[0].tolist()]}")
print()


class BertStyleClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_heads=4, num_layers=2, max_len=32, num_classes=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab["[PAD]"])
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_repr = self.norm(x[:, 0, :])
        return self.classifier(cls_repr), cls_repr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertStyleClassifier(vocab_size=len(vocab)).to(device)
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)
labels = labels.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("【第二部分】模型结构")
print("-" * 80)
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print()

print("【第三部分】训练")
print("-" * 80)
for epoch in range(1, 41):
    model.train()
    logits, cls_repr = model(input_ids, attention_masks)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 10 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        print(f"Epoch {epoch:2d}: loss={loss.item():.4f}, acc={acc:.2%}, cls_norm={cls_repr.norm(dim=1).mean().item():.4f}")
print()

print("【第四部分】预测结果")
print("-" * 80)
model.eval()
with torch.no_grad():
    logits, cls_repr = model(input_ids, attention_masks)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

for idx, (text, label) in enumerate(samples):
    pred = preds[idx].item()
    prob = probs[idx, pred].item()
    print(
        f"{idx + 1:2d}. label={label} pred={pred} prob={prob:.2%} | {text}"
    )
print()

print("【第五部分】[CLS] 表示分析")
print("-" * 80)
cls_cpu = cls_repr.cpu()
for idx in range(min(3, cls_cpu.size(0))):
    print(f"样本 {idx + 1} 的 [CLS] 前 8 维: {cls_cpu[idx, :8]}")
print()

print("【学习总结】")
print("-" * 80)
print(
    """
核心结论：
  ✓ BERT 的主体是 Transformer Encoder 堆叠
  ✓ [CLS] token 负责汇总整句表示，常用于分类任务
  ✓ Attention Mask 用于屏蔽 [PAD] 位置
  ✓ 微调分类任务时，通常在 [CLS] 上接线性层即可

如果之后安装 transformers：
  pip install transformers tokenizers
  然后可切换到 AutoTokenizer + AutoModel 进行真实微调
"""
)
print("=" * 80)