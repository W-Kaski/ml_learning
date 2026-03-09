#!/usr/bin/env python3
"""
文本情感分析 - 双向 BiLSTM + 注意力
=================================

学习内容：
- 文本处理与词汇表构造
- 词嵌入 (Word Embeddings)
- 双向 LSTM (BiLSTM) 编码
- 简单注意力机制 (Attention)
- 文本分类流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

print("=" * 80)
print("【文本情感分析】BiLSTM + 注意力机制")
print("=" * 80)
print()

# ============================================================================
# 【第一部分】文本数据处理与词汇表
# ============================================================================
print("【第一部分】文本数据处理与词汇化")
print("-" * 80)

# 模拟推文/评论数据
texts = [
    "I love this movie it is amazing",
    "This film is terrible and boring",
    "Great performances by all actors",
    "Worst movie I have ever seen",
    "Absolutely fantastic this is brilliant",
    "Bad acting and poor direction",
    "Simply wonderful and heartwarming",
    "Disappointing and a waste of time",
    "This is the best film ever made",
    "Horrible screenplay and terrible plot",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=正面, 0=负面

print(f"样本数: {len(texts)}")
print(f"每个样本是一条评论:")
for i in range(3):
    label_text = "正面" if labels[i] == 1 else "负面"
    print(f"  {i+1}. [{label_text}] {texts[i][:50]}...")
print()

# 文本预处理
def preprocess_text(text):
    """简单的文本清理"""
    text = text.lower()
    words = text.split()
    return words

# 构造词汇表
vocab = Counter()
for text in texts:
    words = preprocess_text(text)
    vocab.update(words)

# 添加特殊令牌
word2idx = {'<PAD>': 0, '<UNK>': 1}
for word, _ in vocab.most_common():
    if word not in word2idx:
        word2idx[word] = len(word2idx)

idx2word = {idx: word for word, idx in word2idx.items()}

print(f"词汇表大小: {len(word2idx)}")
print(f"特殊令牌: PAD={word2idx['<PAD>']}, UNK={word2idx['<UNK>']}")
print(f"最常见的 8 个词:")
for word, count in vocab.most_common(8):
    print(f"  {word:15s}: {count:3d} 次 (idx={word2idx[word]})")
print()

# ============================================================================
# 【第二部分】句子向量化与填充
# ============================================================================
print("【第二部分】句子向量化与批处理")
print("-" * 80)

def text_to_sequence(text, word2idx, max_len=None):
    """将文本转换为索引序列"""
    words = preprocess_text(text)
    sequence = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    return sequence

def pad_sequences(sequences, max_len, pad_idx=0):
    """将序列填充到固定长度"""
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))
    return padded

# 定义最大句子长度
max_len = 12

# 转换所有句子
sequences = [text_to_sequence(text, word2idx) for text in texts]

print(f"最大句子长度: {max_len}")
print(f"原始句子长度分布:")
lengths = [len(seq) for seq in sequences]
print(f"  - 最短: {min(lengths)} 个词")
print(f"  - 最长: {max(lengths)} 个词")
print(f"  - 平均: {np.mean(lengths):.1f} 个词")
print()

# 示例：查看第一条评论的向量化
print(f"示例：第一条评论的向量化")
print(f"  原文: {texts[0]}")
print(f"  序列: {sequences[0]}")
print(f"  词语: {[idx2word[idx] for idx in sequences[0]]}")
print()

# 填充序列
padded_sequences = pad_sequences(sequences, max_len)
X_tensor = torch.tensor(padded_sequences, dtype=torch.long)  # (num_samples, max_len)
y_tensor = torch.tensor(labels, dtype=torch.long)  # (num_samples,)

print(f"张量形状:")
print(f"  - X: {X_tensor.shape} (batch_size, seq_len)")
print(f"  - y: {y_tensor.shape}")
print(f"  - 示例 X[0]: {X_tensor[0]}")
print()

# ============================================================================
# 【第三部分】词嵌入的直观理解
# ============================================================================
print("【第三部分】词嵌入空间")
print("-" * 80)

embedding_dim = 8
embedding = nn.Embedding(len(word2idx), embedding_dim, padding_idx=0)

print(f"词嵌入配置:")
print(f"  - 词汇表大小: {len(word2idx)}")
print(f"  - 嵌入维度: {embedding_dim}")
print(f"  - 参数量: {len(word2idx) * embedding_dim:,}")
print()

# 获取几个词的嵌入
words_to_embed = ['love', 'terrible', 'movie']
print(f"词嵌入示例 (维度={embedding_dim}):")
for word in words_to_embed:
    if word in word2idx:
        idx = word2idx[word]
        emb = embedding(torch.tensor([idx])).squeeze().detach()
        print(f"  '{word:10s}' (idx={idx:2d}): {emb[:4]}... (显示前 4 维)")
print()

# ============================================================================
# 【第四部分】BiLSTM 编码器
# ============================================================================
print("【第四部分】双向 LSTM 编码器与注意力")
print("-" * 80)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                             bidirectional=True, dropout=0.1)
        # 注意力层（简单加性注意力）
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 双向 → 2*hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (B, seq_len, embedding_dim)
        
        # BiLSTM 编码
        lstm_out, (h_n, c_n) = self.bilstm(embedded)  # lstm_out: (B, seq_len, 2*hidden)
        
        # 简单注意力权重
        attention_scores = self.attention(lstm_out)  # (B, seq_len, 1)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, seq_len, 1)
        
        # 带权平均
        context = (lstm_out * attention_weights).sum(dim=1)  # (B, 2*hidden)
        
        # 分类
        logits = self.fc(context)  # (B, output_dim)
        return logits, attention_weights.squeeze(-1)

# 创建模型
vocab_size = len(word2idx)
embedding_dim = 8
hidden_dim = 16
output_dim = 2  # 二分类

model = BiLSTMAttention(vocab_size, embedding_dim, hidden_dim, output_dim)
params = sum(p.numel() for p in model.parameters())

print(f"模型架构:")
print(f"  - Embedding: {vocab_size} × {embedding_dim}")
print(f"  - BiLSTM: {embedding_dim} → {hidden_dim} (双向 → {hidden_dim*2})")
print(f"  - 注意力: 加性注意力 ({hidden_dim*2} → 1)")
print(f"  - 分类层: {hidden_dim*2} → {output_dim}")
print(f"  - 总参数量: {params:,}")
print()

# ============================================================================
# 【第五部分】构造掩码 (Mask) - 处理填充
# ============================================================================
print("【第五部分】处理填充令牌")
print("-" * 80)

def create_mask(X, pad_idx=0):
    """创建掩码，标记填充位置"""
    mask = (X == pad_idx)  # True 表示填充位置
    return mask

mask = create_mask(X_tensor)
print(f"掩码形状: {mask.shape}")
print(f"填充令牌数: {mask.sum().item()}/{mask.numel()}")
print(f"示例 (第一条评论):")
print(f"  - 原始序列: {X_tensor[0]}")
print(f"  - 掩码:     {mask[0].int()}")
print()

# ============================================================================
# 【第六部分】训练
# ============================================================================
print("【第六部分】模型训练")
print("-" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
mask = mask.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

print(f"训练配置:")
print(f"  - 设备: {device}")
print(f"  - Epochs: {num_epochs}")
print(f"  - 优化器: Adam (lr=0.001)")
print(f"  - 损失函数: CrossEntropyLoss")
print()

print(f"【训练进度】")
for epoch in range(num_epochs):
    model.train()
    logits, attn_weights = model(X_tensor, mask)
    loss = criterion(logits, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y_tensor).float().mean()
        print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss={loss.item():.4f}, Accuracy={acc:.2%}")

print()

# ============================================================================
# 【第七部分】模型评估与预测
# ============================================================================
print("【第七部分】模型预测与注意力分析")
print("-" * 80)

model.eval()
with torch.no_grad():
    logits, attn_weights = model(X_tensor, mask)
    preds = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)
    
    accuracy = (preds == y_tensor).float().mean()
    
print(f"整体准确率: {accuracy:.2%}")
print()

print(f"【预测结果示例】")
print(f"{'No.':>3} | {'标签':>4} | {'预测':>4} | {'概率':>8} | {'评论':>40}")
print("-" * 75)

for i in range(min(5, len(texts))):
    label = labels[i]
    pred = preds[i].item()
    prob = probs[i, pred].item()
    label_str = "正" if label == 1 else "负"
    pred_str = "正" if pred == 1 else "负"
    correct = "✓" if pred == label else "✗"
    review = texts[i][:35] + "..." if len(texts[i]) > 35 else texts[i]
    
    print(f"{i+1:3d} | {label_str:>4} | {pred_str:>4} | {prob:7.2%} | {review:>40} {correct}")

print()

# ============================================================================
# 【第八部分】注意力可视化
# ============================================================================
print("【第八部分】注意力权重可视化")
print("-" * 80)

print(f"【注意力权重（前 3 个样本）】")
print()

for i in range(min(3, len(texts))):
    words = preprocess_text(texts[i])
    padded_words = words + ['<PAD>'] * (max_len - len(words))
    
    weights = attn_weights[i].cpu().numpy()
    
    print(f"样本 {i+1}: [{('正面' if labels[i] == 1 else '负面')}]")
    print(f"词语: {padded_words}")
    
    # 显示注意力权重柱状图
    print(f"权重: ", end="")
    for j, (word, weight) in enumerate(zip(padded_words, weights)):
        if j < len(words):  # 非填充词
            bar_len = int(weight * 20)
            bar = "█" * bar_len
            print(f"\n  {word:12s} {weight:6.2%} {bar}", end="")
        else:
            print(f"\n  {word:12s} {weight:6.2%} (PAD)", end="")
    
    print("\n")

# ============================================================================
# 【第九部分】词级别重要性
# ============================================================================
print("【第九部分】模型学到的词级别重要性")
print("-" * 80)

# 分析各个词的嵌入向量与情感的关系
model.eval()
with torch.no_grad():
    # 计算每个词在正面/负面预测中的贡献
    word_contrib = {}
    
    for sample_idx in range(len(texts)):
        sequence = sequences[sample_idx]
        weights = attn_weights[sample_idx].cpu().numpy()
        
        for pos, (word_idx, weight) in enumerate(zip(sequence, weights[:len(sequence)])):
            word = idx2word.get(word_idx, '<UNK>')
            if word not in ['<PAD>', '<UNK>']:
                if word not in word_contrib:
                    word_contrib[word] = []
                # 记录词的注意力权重与标签
                word_contrib[word].append((weight, labels[sample_idx]))
    
    # 计算每个词平均的注意力权重
    word_avg_weight = {word: np.mean([w for w, _ in contribs]) 
                       for word, contribs in word_contrib.items()}

print(f"词的平均注意力权重（按重要性排序）:")
for word, weight in sorted(word_avg_weight.items(), key=lambda x: x[1], reverse=True)[:10]:
    bar = "█" * int(weight * 30)
    print(f"  '{word:12s}': {weight:6.2%} {bar}")

print()

# ============================================================================
# 【总结】
# ============================================================================
print("=" * 80)
print("【学习总结】")
print("=" * 80)
print("""
核心概念：
  ✓ 词嵌入(Embedding): 将离散词汇映射到连续向量空间
  ✓ BiLSTM: 正向+反向编码，捕捉全局上下文
  ✓ 注意力机制: 动态加权，突出重要词汇
  ✓ 顺序池化(Attention Pooling): 比最后隐状态更优
  
NLP 处理最佳实践：
  ✓ 预处理: 小写、分词、去标点
  ✓ 词汇表管理: 特殊令牌 (<PAD>, <UNK>)
  ✓ 固定序列长度: 填充或截断
  ✓ 掩码处理: 防止注意力关注填充
  
常用模型组件：
  ✓ 预训练嵌入(GloVe, FastText): 性能提升 10-20%
  ✓ 层规范化(LayerNorm): 训练稳定性
  ✓ 正则化(Dropout): 过拟合防止
  ✓ 多头注意力: 捕捉不同的语义关系
  
下一步：
  → 使用预训练嵌入 (GloVe, Word2Vec)
  → 学习 Transformer 用于文本处理 (BERT, GPT)
  → 尝试多任务学习 (分类 + 信息抽取)
""")
print("=" * 80)
