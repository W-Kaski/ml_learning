#!/usr/bin/env python3
"""
02_text_classification/data_processor.py
词表构建、文本编码、DataLoader 构建
"""

import re
import random
from collections import Counter
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 第一节：文本预处理
# ============================================================

PAD_IDX = 0
UNK_IDX = 1


def normalize_text(text: str) -> str:
    """转小写、去除多余空格和标点"""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


# ============================================================
# 第二节：词表（Vocabulary）
# ============================================================

class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2idx: Dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2token: Dict[int, str] = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}

    def build(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        return self

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = tokenize(text)[:max_length]
        ids = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        # 补 PAD 至 max_length
        ids += [PAD_IDX] * (max_length - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)


# ============================================================
# 第三节：合成情感数据集
# ============================================================

# 简单的情感词典——不依赖任何外部下载
_POS_WORDS = [
    "good", "great", "excellent", "wonderful", "amazing", "love", "best",
    "fantastic", "brilliant", "superb", "happy", "enjoy", "perfect", "nice",
    "beautiful", "awesome", "splendid", "glorious", "delightful", "pleased"
]
_NEG_WORDS = [
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "dreadful", "disgusting", "boring", "disappointing", "ugly", "mediocre",
    "lousy", "pathetic", "stupid", "waste", "useless", "annoying", "dull"
]
_FILLERS = [
    "the", "this", "was", "is", "very", "really", "so", "quite", "a", "an",
    "movie", "film", "book", "product", "experience", "show", "story"
]


def _make_sentence(sentiment: int, rng: random.Random) -> str:
    fillers = [rng.choice(_FILLERS) for _ in range(rng.randint(2, 5))]
    if sentiment == 1:  # positive
        words = [rng.choice(_POS_WORDS) for _ in range(rng.randint(1, 3))]
    else:               # negative
        words = [rng.choice(_NEG_WORDS) for _ in range(rng.randint(1, 3))]
    tokens = fillers + words
    rng.shuffle(tokens)
    return " ".join(tokens)


def build_synthetic_data(
    total: int = 1000,
    seed: int = 42
) -> List[Tuple[str, int]]:
    """生成 total 条合成情感样本，正负各半。"""
    rng = random.Random(seed)
    data = []
    for i in range(total):
        label = i % 2  # 交替 0/1
        data.append((_make_sentence(label, rng), label))
    rng.shuffle(data)
    return data


# ============================================================
# 第四节：PyTorch Dataset
# ============================================================

class TextDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], vocab: Vocabulary, max_length: int):
        self.samples = samples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        token_ids = self.vocab.encode(text, self.max_length)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============================================================
# 第五节：构建 DataLoader
# ============================================================

def build_dataloaders(config):
    """返回 (train_loader, val_loader, test_loader, vocab)"""
    total = config.train_size + config.val_size + config.test_size
    data = build_synthetic_data(total=total, seed=config.seed)

    # 先用训练集文本构建词表
    train_texts = [t for t, _ in data[:config.train_size]]
    vocab = Vocabulary(min_freq=1).build(train_texts)

    # 构建三个 Dataset
    train_ds = TextDataset(data[:config.train_size], vocab, config.max_length)
    val_ds   = TextDataset(data[config.train_size : config.train_size + config.val_size], vocab, config.max_length)
    test_ds  = TextDataset(data[config.train_size + config.val_size :], vocab, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, vocab