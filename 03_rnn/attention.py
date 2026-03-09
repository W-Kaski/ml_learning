#!/usr/bin/env python3
"""
注意力机制 (Attention) - 深度解析
==============================

学习内容：
- 注意力的数学原理 (加性注意力、乘性注意力)
- 带注意力的 Seq2Seq
- 注意力可视化
- 多头注意力 (Multi-Head Attention) 预备
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 80)
print("【注意力机制(Attention)】从基础到应用")
print("=" * 80)
print()

# ============================================================================
# 【第一部分】为什么需要注意力机制
# ============================================================================
print("【第一部分】注意力的动机")
print("-" * 80)

print("""
问题：固定长度上下文向量的瓶颈

【不用注意力的 Seq2Seq】：
  源序列: "我是一个学生"  (4 个词)
  ↓ Encoder (LSTM)
  上下文向量 C: [0.5, -0.3, 0.8, ...]  (固定长度，e.g. 16维)
  ↓ Decoder (LSTM)
  输出: "I am a student"

问题：
  1. 所有源词的信息被压缩为一个向量 C
  2. 长序列时信息损失严重 (马尔可夫假设内存有界)
  3. 每个目标词的生成，都使用同一个 C (不够灵活)

【有注意力的 Seq2Seq】：
  源序列: "我是一个学生"
  ↓ Encoder 保有所有时间步的隐状态: [h1, h2, h3, h4]
  生成目标词时，动态地关注相关的源词：
    - "I" ← 关注 [h1]    (权重: [0.8, 0.1, 0.05, 0.05])
    - "am" ← 关注 [h1, h2] (权重: [0.3, 0.5, 0.1, 0.1])
    - "student" ← 关注 [h4]  (权重: [0.05, 0.05, 0.1, 0.8])

优势：
  ✓ 信息保存完整: 所有源词信息都保有
  ✓ 对齐对应: 每个目标词可关注最相关的源词
  ✓ 可解释性: 注意力权重可视化显示输入-输出对应关系
""")
print()

# ============================================================================
# 【第二部分】加性注意力 (Additive Attention / Bahdanau)
# ============================================================================
print("【第二部分】加性注意力的数学原理")
print("-" * 80)

# 加性注意力公式：
# score(query, key) = v^T * tanh(W_q * query + W_k * key)

print("""
加性注意力 (Bahdanau Attention):

计算流程：
  1. 准备查询 q (解码器隐状态) 和 键 k (编码器隐状态)
  2. 计算相似度: score = v^T * tanh(W_q*q + W_k*k)
  3. 归一化权重: alpha = softmax(score)
  4. 加权求和: context = sum(alpha * value)
  
公式：
  score_ij = v^T * tanh([W_q * q_i; W_k * k_j])
  alpha_j = exp(score_ij) / sum_k(exp(score_ik))
  context = sum_j(alpha_j * v_j)
  
其中：
  q_i: 解码器时间步 i 的隐状态 (query)
  k_j: 编码器时间步 j 的隐状态 (key)
  v_j: 编码器时间步 j 的隐状态 (value，可以不同)
  W_q, W_k: 可学习的权重矩阵
  v: 可学习的向量

参数复杂度：O(hidden_dim^2)
计算复杂度：O(src_len * hidden_dim)
""")
print()

# 实现加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        nn.init.uniform_(self.v, -0.1, 0.1)
    
    def forward(self, query, keys, mask=None):
        """
        query: (batch_size, hidden_dim) - 解码器隐状态
        keys: (batch_size, src_len, hidden_dim) - 编码器所有隐状态
        mask: (batch_size, src_len) - 填充掩码
        """
        # query 投影: (B, hidden_dim) → (B, hidden_dim)
        query_proj = self.query_proj(query).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # keys 投影: (B, src_len, hidden_dim) → (B, src_len, hidden_dim)
        keys_proj = self.key_proj(keys)  # (B, src_len, hidden_dim)
        
        # 计算评分
        combined = torch.tanh(query_proj + keys_proj)  # (B, src_len, hidden_dim)
        scores = torch.matmul(combined, self.v)  # (B, src_len)
        
        # 掩码处理
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        # 归一化权重
        weights = F.softmax(scores, dim=1)  # (B, src_len)
        
        # 加权求和
        context = torch.einsum('bs,bsh->bh', weights, keys)  # (B, hidden_dim)
        
        return context, weights

# ============================================================================
# 【第三部分】乘性注意力 (Multiplicative Attention / Luong)
# ============================================================================
print("【第三部分】乘性注意力")
print("-" * 80)

# MPI (Multiplicative) Attention：也称为点积注意力
# score(query, key) = query^T * W * key

print("""
乘性注意力 (Luong Attention):

公式：
  score_ij = q_i^T * W * k_j
  alpha_j = exp(score_ij) / sum_k(exp(score_ik))
  context = sum_j(alpha_j * v_j)

特点：
  ✓ 参数少: 只有一个 W 矩阵
  ✓ 计算快: 可以矩阵化为 Q^T * K → (B, query_len, key_len)
  ✓ 推理快: 单位复杂度的注意力计算
  
注意：
  - query 和 key 维度必须相等！
  - 适合维度不大时使用
""")
print()

class MultiplicativeAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, query, keys, mask=None):
        """
        query: (batch_size, hidden_dim)
        keys: (batch_size, src_len, hidden_dim)
        """
        # query 投影: (B, hidden_dim) → (B, hidden_dim)
        query_proj = self.W(query)  # (B, hidden_dim)
        
        # 计算评分：Q * K^T
        scores = torch.matmul(keys, query_proj.unsqueeze(-1)).squeeze(-1)  # (B, src_len)
        
        # 掩码处理
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        # 归一化权重
        weights = F.softmax(scores, dim=1)  # (B, src_len)
        
        # 加权求和
        context = torch.einsum('bs,bsh->bh', weights, keys)  # (B, hidden_dim)
        
        return context, weights

# ============================================================================
# 【第四部分】手工对比实验
# ============================================================================
print("【第四部分】手工对比两种注意力")
print("-" * 80)

# 设置数据
batch_size = 3
src_len = 5
query_dim = 8
hidden_dim = 8

torch.manual_seed(42)

# 生成模拟的编码器隐状态和解码器隐状态
encoder_hidden = torch.randn(batch_size, src_len, hidden_dim)  # (B, src_len, hidden)
decoder_hidden = torch.randn(batch_size, query_dim)  # (B, hidden)

# 创建两种注意力层
add_attn = AdditiveAttention(hidden_dim)
mul_attn = MultiplicativeAttention(hidden_dim)

print(f"输入形状:")
print(f"  - 编码器隐状态 (所有时间步): {encoder_hidden.shape}")
print(f"  - 解码器隐状态 (当前步): {decoder_hidden.shape}")
print()

# 计算注意力
context_add, weights_add = add_attn(decoder_hidden, encoder_hidden)
context_mul, weights_mul = mul_attn(decoder_hidden, encoder_hidden)

print(f"注意力输出:")
print(f"  - 加性注意力 Context: {context_add.shape}, 权重: {weights_add.shape}")
print(f"  - 乘性注意力 Context: {context_mul.shape}, 权重: {weights_mul.shape}")
print()

print(f"注意力权重分布 (示例：第一个样本的三个时间步):")
print(f"{'':>8} | {'加性注意力':^30} | {'乘性注意力':^30}")
print(f"{'时间步':>8} | {'权重':>10} | {'分布':>17} | {'权重':>10} | {'分布':>17}")
print("-" * 80)

for t in range(src_len):
    add_w = weights_add[0, t].item()
    mul_w = weights_mul[0, t].item()
    add_bar = "█" * int(add_w * 20)
    mul_bar = "█" * int(mul_w * 20)
    print(f"{'t=' + str(t):>8} | {add_w:10.4f} | {add_bar:>17} | {mul_w:10.4f} | {mul_bar:>17}")

print()

# ============================================================================
# 【第五部分】缩放点积注意力 (Scaled Dot-Product Attention)
# ============================================================================
print("【第五部分】缩放点积注意力 (Transformer 基础)")
print("-" * 80)

print(f"""
缩放点积注意力（这是 Transformer 中的标准注意力）：

公式：
  Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V
  
其中：
  Q (Query): (batch_size, seq_len, d_k)
  K (Key): (batch_size, seq_len, d_k)
  V (Value): (batch_size, seq_len, d_v)
  d_k: key 的维度
    
缩放因子 √d_k 的作用：
  - 未缩放时：Q*K^T 的数值会很大 (∝ d_k)
  - 大的值送入 softmax 后，会产生极端的权重 (接近 0/1)
  - 这导致梯度消失
  - 缩放后，方差恢复，梯度流更好
  
复杂度：
  - 参数: 0 (无参数)
  - 计算: O(seq_len^2 * d_k)
""")
print()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, query, key, value, mask=None):
        """
        query: (batch_size, seq_len_q, d_k)
        key: (batch_size, seq_len_k, d_k)
        value: (batch_size, seq_len_v, d_v)
        """
        # 计算评分
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        
        # 权重
        weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(weights, value)
        
        return context, weights

# 测试缩放点积注意力
scaled_attn = ScaledDotProductAttention(hidden_dim)

# 模拟多头注意力的数据格式：(B, num_heads, seq_len, d_k)
B, num_heads, seq_len, d_k = 2, 4, 6, 8

Q = torch.randn(B, num_heads, seq_len, d_k)
K = torch.randn(B, num_heads, seq_len, d_k)
V = torch.randn(B, num_heads, seq_len, d_k)

context_scaled, weights_scaled = scaled_attn(Q, K, V)

print(f"缩放点积注意力演示:")
print(f"  - Query 形状: {Q.shape}")
print(f"  - Key 形状: {K.shape}")
print(f"  - Value 形状: {V.shape}")
print(f"  - Context 输出: {context_scaled.shape}")
print(f"  - 注意力权重: {weights_scaled.shape}")
print()

# 分析权重分布
weights_mean = weights_scaled.mean()
weights_std = weights_scaled.std()
weights_max = weights_scaled.max()
weights_min = weights_scaled.min()

print(f"注意力权重统计:")
print(f"  - 均值: {weights_mean:.4f}")
print(f"  - 标准差: {weights_std:.4f}")
print(f"  - 最大值: {weights_max:.4f}")
print(f"  - 最小值: {weights_min:.4f}")
print(f"  - 熵: {-(weights_scaled * torch.log(weights_scaled + 1e-9)).mean():.4f}")
print()

# ============================================================================
# 【第六部分】注意力矩阵可视化
# ============================================================================
print("【第六部分】注意力权重矩阵分析")
print("-" * 80)

# 模拟一个简单的对齐问题
# 源语言: "hello world how are you"
# 目标语言: "你好 世界 怎么样 呢"

src_len = 5
tgt_len = 4

# 构造理想的注意力权重 (真实对齐)
ideal_attention = torch.tensor([
    [0.7, 0.3, 0.0, 0.0, 0.0],  # "你好" ← "hello"
    [0.0, 0.9, 0.1, 0.0, 0.0],  # "世界" ← "world"
    [0.0, 0.0, 0.6, 0.4, 0.0],  # "怎么样" ← "how are"
    [0.0, 0.0, 0.0, 0.1, 0.9],  # "呢" ← "you"
])

src_words = ["hello", "world", "how", "are", "you"]
tgt_words = ["你好", "世界", "怎么样", "呢"]

print(f"理想的注意力对齐矩阵:")
print()
print(f"           ", end="")
for src_word in src_words:
    print(f"{src_word:>8} ", end="")
print()
print("-" * 48)

for tgt_idx, tgt_word in enumerate(tgt_words):
    print(f"{tgt_word:>8} | ", end="")
    for src_idx in range(src_len):
        weight = ideal_attention[tgt_idx, src_idx].item()
        bar = "█" * int(weight * 8)
        print(f"{weight:6.1%} {bar:>1} | ", end="")
    print()

print()
print("观察:")
print(f"  ✓ 对角线清晰: 每个目标词关注对应的源词")
print(f"  ✓ 稀疏分布: 权重集中在少数几个源词")
print(f"  ✓ 单调性: 注意力焦点从左向右移动")
print()

# ============================================================================
# 【第七部分】实际应用示例
# ============================================================================
print("【第七部分】Seq2Seq + 注意力完整例子")
print("-" * 80)

class AttentionSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        
        self.attention = ScaledDotProductAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        # 编码
        src_emb = self.encoder_embedding(src)
        encoder_out, (h_n, c_n) = self.encoder(src_emb)
        
        # 解码（简化版：单步演示）
        tgt_emb = self.decoder_embedding(tgt[:, 0:1])
        
        # 注意力
        context, attn_weights = self.attention(
            h_n.transpose(0, 1),  # (B, 1, hidden) - 查询
            encoder_out,           # (B, src_len, hidden) - 键值
            encoder_out
        )
        
        # 解码器输入 = 嵌入 + 上下文
        decoder_input = torch.cat([tgt_emb, context], dim=-1)
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))
        
        logits = self.fc(decoder_out)
        return logits, attn_weights

# 创建模型并演示
model_attn = AttentionSeq2Seq(
    src_vocab_size=50,
    tgt_vocab_size=50,
    embedding_dim=8,
    hidden_dim=16
)

print(f"带注意力的 Seq2Seq 模型:")
print(f"  - 编码器：将输入序列编码为所有时间步的隐状态")
print(f"  - 注意力：在每一步动态选择最相关的源信息")
print(f"  - 解码器：利用 [嵌入 + 上下文] 生成输出")
print()

# 模拟输入
src_seq = torch.randint(1, 50, (2, 6))  # (B=2, src_len=6)
tgt_seq = torch.randint(1, 50, (2, 4))  # (B=2, tgt_len=4)

logits, attn_w = model_attn(src_seq, tgt_seq)

print(f"推理输出:")
print(f"  - Logits: {logits.shape}")
print(f"  - 注意力权重: {attn_w.shape}")
print()

# ============================================================================
# 【总结】
# ============================================================================
print("=" * 80)
print("【学习总结】")
print("=" * 80)
print("""
关键概念：
  ✓ 注意力 = 软焦点: 不同位置的不同关注度
  ✓ 三元组 (Q, K, V):
    Query (查询): 当前想问什么问题
    Key (键): 源序列中有什么信息
    Value (值): 返回什么信息
  
三种注意力机制对比：

  | 机制 | 公式 | 参数 | 计算复杂度 | 优点 |
  |-----|------|------|----------|------|
  | 加性 | v^T*tanh(Wq*q+Wk*k) | O(d^2) | O(src*d) | 灵活，性能好 |
  | 乘性 | q^T*W*k | O(d^2) | O(src*d) | 快速，可矩阵化 |
  | 缩放点积 | Q*K^T/√d | O(1) | O(src^2*d) | 无参数，Transformer基础 |
  
应用场景：
  ✓ 机器翻译: 源-目对齐
  ✓ 文本摘要: 重点词突出
  ✓ 视觉问答: 关注图像区域
  ✓ 语音识别: 音频对齐
  
关键改进：
  ✓ 多头注意力: 不同的表示空间关注不同的信息
  ✓ 自注意力: 序列内部的上下文依赖性
  ✓ 交叉注意力: 两个不同模态间的对齐
  
下一步：
  → 学习多头注意力 (Multi-Head Attention)
  → 自注意力与 Transformer 架构
  → 预训练模型 (BERT, T5, GPT)
""")
print("=" * 80)
