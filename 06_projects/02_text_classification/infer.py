#!/usr/bin/env python3
"""
02_text_classification/infer.py

对单条原始文本做情感分类推理：
  - 接受 --text "some sentence" 或 --index N（从测试集取第 N 条）
  - 输出预测类别、置信度、Top-2 概率分布
  - 结果保存至 outputs/infer_result.json

用法：
  python3 infer.py --text "this movie is really great"
  python3 infer.py --index 5 --checkpoint last
"""

import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_processor import (build_synthetic_data, Vocabulary,
                             tokenize, PAD_IDX, UNK_IDX)
from model import SimpleTextClassifier


# ============================================================
# 第一节：推理函数
# ============================================================

LABEL_NAMES = ["negative", "positive"]


def encode_text(text: str, vocab: Vocabulary, max_length: int) -> torch.Tensor:
    """将原始字符串编码为 (1, max_length) 的 token id 张量。"""
    ids = vocab.encode(text, max_length)
    return torch.tensor([ids], dtype=torch.long)


def predict(model, token_ids: torch.Tensor, device: torch.device):
    """返回 (pred_label_idx, confidence, probabilities_list)"""
    model.eval()
    with torch.no_grad():
        logits = model(token_ids.to(device))
        probs = F.softmax(logits, dim=-1)[0].cpu()
    pred = probs.argmax().item()
    return pred, probs[pred].item(), probs.tolist()


# ============================================================
# 第二节：主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, default=None,
                       help="直接输入要分类的文本字符串")
    group.add_argument("--index", type=int, default=None,
                       help="从测试集取第 N 条样本（0-based）")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()

    # 如果既没有 --text 也没有 --index，默认用测试集第 0 条
    if args.text is None and args.index is None:
        args.index = 0

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 确认 checkpoint 存在 ---
    ckpt_path = config.best_ckpt_path if args.checkpoint == "best" else config.last_ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}\n请先运行 train.py")

    # --- 加载词表 ---
    vocab_path = config.output_path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"词表文件不存在: {vocab_path}\n请先运行 train.py")
    with open(vocab_path) as f:
        token2idx = json.load(f)
    vocab = Vocabulary()
    vocab.token2idx = token2idx
    vocab.idx2token = {v: k for k, v in token2idx.items()}
    actual_vocab_size = len(vocab)

    # --- 决定推理文本 ---
    true_label = None
    if args.text is not None:
        input_text = args.text
    else:
        total = config.train_size + config.val_size + config.test_size
        data = build_synthetic_data(total=total, seed=config.seed)
        test_data = data[config.train_size + config.val_size:]
        if args.index >= len(test_data):
            raise IndexError(f"--index {args.index} 超出测试集大小 {len(test_data)}")
        input_text, true_label = test_data[args.index]

    # --- 加载模型 ---
    model = SimpleTextClassifier(
        vocab_size=actual_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --- 推理 ---
    token_ids = encode_text(input_text, vocab, config.max_length)
    pred_idx, confidence, probs = predict(model, token_ids, device)
    pred_label = LABEL_NAMES[pred_idx]

    # --- 输出 ---
    print(f"=== 02_text_classification 推理 ===")
    print(f"输入文本  : {input_text!r}")
    if true_label is not None:
        print(f"真实标签  : {LABEL_NAMES[true_label]} ({true_label})")
    print(f"预测标签  : {pred_label} ({pred_idx})")
    print(f"置信度    : {confidence * 100:.2f}%")
    print(f"概率分布  :")
    for i, p in enumerate(probs):
        marker = " <-- 预测" if i == pred_idx else ""
        print(f"  {LABEL_NAMES[i]:10s}: {p * 100:.2f}%{marker}")

    # --- 保存 ---
    result = {
        "input_text": input_text,
        "true_label": true_label,
        "pred_label": pred_label,
        "pred_idx": pred_idx,
        "confidence": round(confidence, 4),
        "probabilities": {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
        "checkpoint": args.checkpoint,
    }
    out_path = config.output_path / "infer_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n推理结果已保存: {out_path}")


if __name__ == "__main__":
    main()
