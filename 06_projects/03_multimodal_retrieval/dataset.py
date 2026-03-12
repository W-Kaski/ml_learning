#!/usr/bin/env python3
"""
03_multimodal_retrieval/dataset.py

合成图文对数据集：
  - 图像：随机噪声 (FakeData 风格)，按类别分配固定的颜色偏置，
           使不同类别的图像在统计上可区分。
  - 文本：由类别词 + 固定模板生成的句子，编码为 token id 序列。
  - 配对：同一类别的 (image, text) 构成正样本对。
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict

# ============================================================
# 第一节：类别词汇表
# ============================================================

# 10 个可区分的语义类别及其描述词
CLASS_NAMES = [
    "cat", "dog", "car", "bird", "house",
    "tree", "boat", "plane", "flower", "fish",
]

TEMPLATES = [
    "a photo of a {cls}",
    "an image showing a {cls}",
    "this is a {cls}",
    "picture of {cls}",
    "{cls} captured in photo",
]

# ============================================================
# 第二节：词表构建
# ============================================================

PAD_IDX = 0
UNK_IDX = 1


def build_vocab() -> Dict[str, int]:
    """从模板 + 类别名构建词表。"""
    token2idx: Dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    fixed_words = [
        "a", "an", "photo", "of", "image", "showing", "this",
        "is", "picture", "captured", "in",
    ]
    all_tokens = fixed_words + CLASS_NAMES
    for token in all_tokens:
        if token not in token2idx:
            token2idx[token] = len(token2idx)
    return token2idx


def encode_text(text: str, token2idx: Dict[str, int], max_length: int) -> List[int]:
    tokens = text.lower().split()[:max_length]
    ids = [token2idx.get(t, UNK_IDX) for t in tokens]
    ids += [PAD_IDX] * (max_length - len(ids))
    return ids


# ============================================================
# 第三节：合成图像生成
# ============================================================

# 为每个类别指定一个颜色偏置（RGB），使图像在统计上可区分
_CLASS_BIAS = torch.linspace(-0.5, 0.5, len(CLASS_NAMES))


def make_image(class_idx: int, channels: int, size: int, rng: torch.Generator) -> torch.Tensor:
    """
    生成带类别偏置的随机图像 (C, H, W)，值域 [-1, 1]。
    - 基础图像为均匀随机噪声
    - 加上与类别相关的颜色偏置，让不同类别的图像均值不同
    """
    img = torch.rand(channels, size, size, generator=rng) * 2 - 1
    bias = _CLASS_BIAS[class_idx]
    img = (img + bias).clamp(-1, 1)
    return img


# ============================================================
# 第四节：Dataset 类
# ============================================================

class MultimodalPairDataset(Dataset):
    """
    每条样本 = (image_tensor, text_token_ids, class_label)。
    同一类别的图文构成正确配对；训练时用 class_label 构造对比损失。
    """

    def __init__(
        self,
        pairs: List[Tuple[torch.Tensor, List[int], int]],
    ):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image, text_ids, label = self.pairs[idx]
        return (
            image,
            torch.tensor(text_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# ============================================================
# 第五节：数据构建入口
# ============================================================

def build_pairs(
    num_per_class: int,
    num_classes: int,
    channels: int,
    size: int,
    max_length: int,
    seed: int,
    token2idx: Dict[str, int],
) -> List[Tuple[torch.Tensor, List[int], int]]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    py_rng = random.Random(seed)

    pairs = []
    for cls_idx in range(num_classes):
        cls_name = CLASS_NAMES[cls_idx % len(CLASS_NAMES)]
        for _ in range(num_per_class):
            img = make_image(cls_idx, channels, size, rng)
            template = py_rng.choice(TEMPLATES)
            text = template.format(cls=cls_name)
            text_ids = encode_text(text, token2idx, max_length)
            pairs.append((img, text_ids, cls_idx))

    py_rng.shuffle(pairs)
    return pairs


def build_dataloaders(config):
    """返回 (train_loader, val_loader, test_loader, token2idx)"""
    token2idx = build_vocab()

    all_pairs = build_pairs(
        num_per_class=config.num_pairs,
        num_classes=config.num_classes,
        channels=config.image_channels,
        size=config.image_size,
        max_length=config.text_max_length,
        seed=config.seed,
        token2idx=token2idx,
    )

    total = len(all_pairs)
    val_size  = max(1, int(total * 0.10))
    test_size = max(1, int(total * 0.10))
    train_size = total - val_size - test_size

    g = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds, test_ds = random_split(
        MultimodalPairDataset(all_pairs),
        [train_size, val_size, test_size],
        generator=g,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, pin_memory=pin)

    return train_loader, val_loader, test_loader, token2idx