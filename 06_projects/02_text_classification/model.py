#!/usr/bin/env python3
"""
02_text_classification/model.py
GRU 文本分类模型
架构：Embedding → Dropout → GRU → Dropout → Linear
"""

import torch
import torch.nn as nn


class SimpleTextClassifier(nn.Module):
    """
    基于双向 GRU 的文本分类器。

    前向传播流程：
      1. Embedding：将 token id 映射为稠密向量，padding_idx=0 不参与梯度
      2. GRU（双向）：提取序列特征，取最后时刻的前向/后向隐向量拼接
      3. Linear：将隐向量映射到类别 logit
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        # bidirectional=True：隐层大小加倍，提升表达能力
        self.encoder = nn.GRU(
            embedding_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.cls_dropout = nn.Dropout(dropout)
        # 双向 GRU 的两个方向拼接，故输入为 hidden_dim * 2
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) — token id 张量
        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.emb_dropout(self.embedding(x))        # (B, L, E)
        _, hidden = self.encoder(embedded)                    # hidden: (2, B, H)
        # 拼接前向(hidden[0])和后向(hidden[1])的最终隐向量
        h = torch.cat([hidden[0], hidden[1]], dim=-1)         # (B, 2H)
        logits = self.head(self.cls_dropout(h))               # (B, num_classes)
        return logits


def build_model(config) -> SimpleTextClassifier:
    """根据 Config 实例化模型，vocab_size 由外部传入。"""
    return SimpleTextClassifier(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    )


if __name__ == "__main__":
    import torch
    model = SimpleTextClassifier()
    x = torch.randint(0, 100, (4, 32))   # batch=4, seq_len=32
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")       # 应为 (4, 2)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
