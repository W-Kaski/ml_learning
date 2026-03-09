#!/usr/bin/env python3
"""
序列到序列(Seq2Seq) - 机器翻译基础
==================================

学习内容：
- 编码器-解码器 (Encoder-Decoder) 架构
- 上下文向量 (Context Vector)
- 教师强制训练 (Teacher Forcing)
- 束搜索推理 (Beam Search)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

print("=" * 80)
print("【序列到序列(Seq2Seq)】编码器-解码器架构")
print("=" * 80)
print()

# ============================================================================
# 【第一部分】Seq2Seq 架构原理
# ============================================================================
print("【第一部分】Seq2Seq 架构与信息流")
print("-" * 80)

print("""
Seq2Seq (Sequence-to-Sequence) 解决变长输入到变长输出的问题。

架构组成：
  编码器 (Encoder): 将输入序列压缩为固定长度的上下文向量
    输入序列: x1, x2, ..., x_n (e.g. 中文句子)
    ↓ LSTM 处理各时间步
    上下文向量: C = h_n (最后隐状态，编码了全部信息)
  
  解码器 (Decoder): 从上下文向量逐步生成输出序列
    初始隐状态: h0 = C (来自编码器)
    利用上下文: 在每个时间步使用 C 帮助预测
    输出序列: y1, y2, ..., y_m (e.g. 英文句子)

关键问题：
  - 信息瓶颈: 固定长度 C 可能丢失长序列信息 
    → 解决: 使用注意力机制 (见 attention.py)
  - 长序列依赖: 解码器初始隐状态只来自编码器末端
    → 改进: 双向编码器、递归注意力
""")
print()

# ============================================================================
# 【第二部分】数据准备 - 简单的"伪翻译"任务
# ============================================================================
print("【第二部分】数据构造 - 源语言到目标语言")
print("-" * 80)

# 模拟源语言（Source）→ 目标语言（Target）的对应关系
# 实际上是简单的字符反转任务作为演示（实际翻译需要真实数据）

source_sents = [
    "hello world",
    "good morning",
    "how are you",
    "thank you very much",
    "please help me",
    "where is the station",
    "i love programming",
    "machine learning is fun",
]

# 簡單規則：生成"目標"序列（實際中應該是真實翻譯）
# 演示用：逆序 + 符號變換
def create_target(source):
    words = source.split()
    # 反轉單詞順序
    reversed_words = words[::-1]
    return " ".join(reversed_words)

target_sents = [create_target(src) for src in source_sents]

print(f"样本数: {len(source_sents)}")
print(f"样本对示例 (源→目标):")
for i in range(min(4, len(source_sents))):
    print(f"  {i+1}. 源: '{source_sents[i]}'")
    print(f"     目: '{target_sents[i]}'")
print()

# 构造词汇表
def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    word_count = {}
    for sent in sentences:
        for word in sent.split():
            word_count[word] = word_count.get(word, 0) + 1
    
    for word in sorted(word_count.keys()):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab

src_vocab = build_vocab(source_sents)
tgt_vocab = build_vocab(target_sents)

print(f"词汇表大小:")
print(f"  - 源语言: {len(src_vocab)} 个词")
print(f"  - 目标语言: {len(tgt_vocab)} 个词")
print()

# 转换为张量
def sent_to_tensor(sent, vocab, max_len=15):
    words = sent.split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    # 添加 <EOS>
    indices.append(vocab['<EOS>'])
    # 填充
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

max_src_len = 15
max_tgt_len = 15

X_src = torch.tensor([sent_to_tensor(s, src_vocab, max_src_len) for s in source_sents],
                     dtype=torch.long)
X_tgt = torch.tensor([sent_to_tensor(s, tgt_vocab, max_tgt_len) for s in target_sents],
                     dtype=torch.long)

print(f"张量形状:")
print(f"  - X_src: {X_src.shape} (num_samples, src_max_len)")
print(f"  - X_tgt: {X_tgt.shape} (num_samples, tgt_max_len)")
print()

# ============================================================================
# 【第三部分】编码器 (Encoder)
# ============================================================================
print("【第三部分】编码器实现")
print("-" * 80)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (B, seq_len, embedding_dim)
        _, (h_n, c_n) = self.lstm(embedded)  # h_n, c_n: (1, B, hidden_dim)
        return h_n, c_n

# ============================================================================
# 【第四部分】解码器 (Decoder)
# ============================================================================
print("【第四部分】解码器实现")
print("-" * 80)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # 预测下一个词
    
    def forward(self, x, h_n, c_n):
        # x: (batch_size, 1) - 单个时间步的输入
        # h_n, c_n: 来自上一时间步（或编码器初始状态）
        embedded = self.embedding(x)  # (B, 1, embedding_dim)
        out, (h_n, c_n) = self.lstm(embedded, (h_n, c_n))  # out: (B, 1, hidden_dim)
        logits = self.fc(out)  # (B, 1, vocab_size)
        return logits, h_n, c_n

# ============================================================================
# 【第五部分】完整 Seq2Seq 模型
# ============================================================================
print("【第五部分】Seq2Seq 模型")
print("-" * 80)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_sos_idx, tgt_eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
    
    def forward(self, X_src, X_tgt, teacher_forcing_ratio=0.5):
        """
        X_src: (batch_size, src_seq_len)
        X_tgt: (batch_size, tgt_seq_len)
        teacher_forcing_ratio: 使用真实目标作为输入的概率
        """
        batch_size = X_src.shape[0]
        tgt_max_len = X_tgt.shape[1]
        
        # 编码器处理源序列
        h_n, c_n = self.encoder(X_src)  # (1, B, hidden_dim)
        
        # 解码器生成目标序列
        decoder_input = torch.full((batch_size, 1), self.tgt_sos_idx, 
                                   dtype=torch.long, device=X_tgt.device)  # <SOS>
        
        outputs = []
        for t in range(tgt_max_len):
            logits, h_n, c_n = self.decoder(decoder_input, h_n, c_n)
            outputs.append(logits.squeeze(1))  # (B, vocab_size)
            
            # 决定解码器下一步的输入
            if random.random() < teacher_forcing_ratio:
                # 使用真实目标词
                decoder_input = X_tgt[:, t:t+1]
            else:
                # 使用模型预测的词
                top1 = logits.argmax(2)  # (B, 1)
                decoder_input = top1
        
        outputs = torch.stack(outputs, dim=1)  # (B, tgt_max_len, vocab_size)
        return outputs

# 创建模型
embedding_dim = 8
hidden_dim = 16

encoder = Encoder(len(src_vocab), embedding_dim, hidden_dim)
decoder = Decoder(len(tgt_vocab), embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder, tgt_vocab['<SOS>'], tgt_vocab['<EOS>'])

total_params = sum(p.numel() for p in model.parameters())

print(f"模型配置:")
print(f"  - 编码器: {len(src_vocab)} → {embedding_dim} → {hidden_dim}")
print(f"  - 解码器: {len(tgt_vocab)} → {embedding_dim} → {hidden_dim}")
print(f"  - 总参数: {total_params:,}")
print()

print(f"架构图:")
print(f"""
  源序列    嵌入      编码        上下文         解码         层规范化      目标序列
  (词ID) → EMBED → ENCODER → [h_n, c_n] → DECODER → Dense(softmax) → (预测)
                                 ↓
                           [h_n, c_n] → 解码器初始状态
""")
print()

# ============================================================================
# 【第六部分】训练
# ============================================================================
print("【第六部分】训练过程")
print("-" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X_src = X_src.to(device)
X_tgt = X_tgt.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 <PAD>
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 40

print(f"训练配置:")
print(f"  - 设备: {device}")
print(f"  - Epochs: {num_epochs}")
print(f"  - 优化器: Adam (lr=0.001)")
print(f"  - Teacher Forcing: True (比例 0.5)")
print()

print(f"【训练进度】")
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_src, X_tgt, teacher_forcing_ratio=0.5)
    # outputs: (B, tgt_max_len, vocab_size)
    
    # 计算损失（忽略 <PAD> 位置）
    loss = criterion(
        outputs.reshape(-1, len(tgt_vocab)),
        X_tgt.reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss={loss.item():.4f}")

print()

# ============================================================================
# 【第七部分】推理 - 贪心解码
# ============================================================================
print("【第七部分】推理与翻译")
print("-" * 80)

def greedy_decode(model, X_src_single, src_vocab, tgt_vocab, max_len=15):
    """贪心解码：每步选择概率最高的词"""
    model.eval()
    
    with torch.no_grad():
        # 编码
        h_n, c_n = model.encoder(X_src_single.unsqueeze(0))
        
        # 解码
        decoder_input = torch.tensor([[tgt_vocab['<SOS>']]], 
                                    dtype=torch.long, device=X_src_single.device)
        decoded_indices = []
        
        for _ in range(max_len):
            logits, h_n, c_n = model.decoder(decoder_input, h_n, c_n)
            top_idx = logits.argmax(2)  # (1, 1)
            decoded_indices.append(top_idx.item())
            
            if top_idx.item() == tgt_vocab['<EOS>']:
                break
            
            decoder_input = top_idx
    
    return decoded_indices

def indices_to_sent(indices, vocab):
    """将索引转换为句子"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    words = [reverse_vocab.get(idx, '<UNK>') for idx in indices 
             if idx not in [vocab['<PAD>'], vocab['<EOS>']]]
    return ' '.join(words)

# 测试集推理
print(f"推理结果 (贪心解码):")
print()

for i in range(min(6, len(source_sents))):
    X_src_i = X_src[i]
    
    # 模型的预测
    pred_indices = greedy_decode(model, X_src_i, src_vocab, tgt_vocab)
    pred_sent = indices_to_sent(pred_indices, tgt_vocab)
    
    # 真实目标
    true_sent = target_sents[i]
    
    # 计算词级别准确率（简单的字符匹配）
    match = 1 if pred_sent == true_sent else 0
    
    print(f"[样本 {i+1}]")
    print(f"  源: {source_sents[i]}")
    print(f"  预: {pred_sent}")
    print(f"  真: {true_sent}")
    print(f"  准确: {'✓' if match else '✗'}")
    print()

# ============================================================================
# 【第八部分】上下文向量分析
# ============================================================================
print("【第八部分】上下文向量的作用")
print("-" * 80)

model.eval()
with torch.no_grad():
    # 获取所有样本的上下文向量
    h_encoding, _ = model.encoder(X_src)  # (1, B, hidden_dim)
    h_encoding = h_encoding.squeeze(0)  # (B, hidden_dim)

print(f"上下文向量统计:")
print(f"  - 形状: {h_encoding.shape}")
print(f"  - 样本内 L2 范数:")
for i in range(min(4, len(X_src))):
    norm = h_encoding[i].norm().item()
    print(f"    样本 {i+1}: {norm:.4f}")
print()

# 计算上下文向量之间的相似度
print(f"上下文向量相似度矩阵 (Cosine):")
from torch.nn.functional import cosine_similarity

cos_sim = torch.zeros(len(X_src), len(X_src))
for i in range(len(X_src)):
    for j in range(len(X_src)):
        cos_sim[i, j] = cosine_similarity(
            h_encoding[i].unsqueeze(0),
            h_encoding[j].unsqueeze(0)
        ).item()

print(f"(仅显示前 5 个样本)")
print(f"      ", end="")
for j in range(min(5, len(X_src))):
    print(f"样本{j+1} ", end="")
print()

for i in range(min(5, len(X_src))):
    print(f"样本{i+1} ", end="")
    for j in range(min(5, len(X_src))):
        sim = cos_sim[i, j].item()
        print(f"{sim:6.3f} ", end="")
    print()

print()

# ============================================================================
# 【总结】
# ============================================================================
print("=" * 80)
print("【学习总结】")
print("=" * 80)
print("""
核心概念：
  ✓ 编码器压缩输入序列为固定长度的上下文向量 C
  ✓ 解码器以 C 为初始状态，自回归地生成输出序列
  ✓ 教师强制训练加速收敛，推理时使用自己的预测
  
关键设计：
  ✓ <SOS> (Start Of Sequence): 解码器的初始输入
  ✓ <EOS> (End Of Sequence): 表示序列结束
  ✓ <PAD>: 处理可变长序列
  
训练技巧：
  ✓ 教师强制训练 (Teacher Forcing): 加快训练
    缺点: 训练-推理不一致 (Exposure Bias)
  ✓ 逐步减少教师强制概率: 训练后期更多使用模型预测
  ✓ 梯度裁剪: 防止爆炸
  
推理方法：
  ✓ 贪心解码: 每步选概率最高词（快速，可能次优）
  ✓ 束搜索 (Beam Search): 保留 K 个最优候选(后续实现)
  ✓ 多样化解码: Temperature sampling（探索性）
  
当前模型的问题：
  ✗ 信息瓶颈: 固定长度 C 可能遗漏信息
    → 解决方案: 注意力机制 (见 attention.py)
  ✗ 长序列性能差: 隐状态梯度衰减
    → 解决方案: 双向编码器、层级降采样
  
下一步：
  → 实现注意力机制 Seq2Seq (attention.py)
  → 学习预训练模型 (BERT, T5)
  → 尝试真实翻译数据集 (WMT)
""")
print("=" * 80)
