#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # 模型超参数
    image_size: int = 32          # 输入图像宽/高（FakeData 使用）
    image_channels: int = 3
    image_feature_dim: int = 128  # CNN 输出特征维度
    text_vocab_size: int = 200    # 合成词表大小（实际由数据决定）
    text_max_length: int = 16     # 句子最大 token 数
    embed_dim: int = 64           # 公共嵌入空间维度

    # 训练超参数
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 15
    temperature: float = 0.07     # InfoNCE 温度参数
    seed: int = 42

    # 数据规模（合成图文对）
    num_pairs: int = 500          # 每类别图文对数量 × num_classes
    num_classes: int = 10         # 语义类别数（决定图文配对难度）

    # 评估
    top_k: int = 5                # Recall@K 中的 K 值

    # 路径
    output_dir: str = "outputs"

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