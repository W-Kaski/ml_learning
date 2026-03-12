#!/usr/bin/env python3

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # 模型超参数
    max_length: int = 32          # 句子最大 token 数（不足补 PAD，超出截断）
    vocab_size: int = 5000        # 词表大小（含 PAD=0, UNK=1）
    embedding_dim: int = 64       # 词向量维度
    hidden_dim: int = 128         # GRU 隐藏层维度
    num_classes: int = 2          # 分类类别数

    # 训练超参数
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    seed: int = 42

    # 数据参数
    train_size: int = 800         # 训练集样本数（合成数据）
    val_size: int = 100
    test_size: int = 100

    # 路径
    output_dir: str = "outputs"

    # ---- 派生路径属性 ----
    @property
    def project_dir(self) -> Path:
        return Path(__file__).parent

    @property
    def output_path(self) -> Path:
        p = self.project_dir / self.output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def best_ckpt_path(self) -> Path:
        return self.output_path / "best_model.pth"

    @property
    def last_ckpt_path(self) -> Path:
        return self.output_path / "last_model.pth"