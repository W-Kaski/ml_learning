#!/usr/bin/env python3
"""
02_text_classification/eval.py

在测试集上评估已训练模型：
  - 整体准确率
  - 各类别 Precision / Recall / F1（手动计算，不依赖 sklearn）
  - 混淆矩阵
  - 错误样本展示（最多 5 条）
  - 输出结果保存至 outputs/eval_metrics.json

用法：
  python3 eval.py                       # 加载 best checkpoint
  python3 eval.py --checkpoint last     # 加载 last checkpoint
"""

import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_processor import build_dataloaders, build_synthetic_data, TextDataset, Vocabulary
from model import SimpleTextClassifier


# ============================================================
# 第一节：评估函数
# ============================================================

def evaluate_full(model, loader, device, num_classes, label_names=None):
    """返回 (accuracy, confusion_matrix, wrong_samples)"""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    wrong = []   # [(true_label, pred_label, sample_idx), ...]

    all_ids = []
    with torch.no_grad():
        sample_offset = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            for i in range(len(y)):
                t, p = y[i].item(), preds[i].item()
                confusion[t][p] += 1
                if t != p and len(wrong) < 5:
                    wrong.append({"true": t, "pred": p, "sample_offset": sample_offset + i})
            sample_offset += len(y)

    total = confusion.sum().item()
    correct = confusion.trace().item()
    accuracy = correct / total if total > 0 else 0.0

    # 计算各类 Precision / Recall / F1
    per_class = []
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        name = label_names[c] if label_names else str(c)
        per_class.append({"class": name, "precision": round(precision, 4),
                           "recall": round(recall, 4), "f1": round(f1, 4)})

    return accuracy, confusion.tolist(), per_class, wrong


# ============================================================
# 第二节：主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 02_text_classification 评估 ===  checkpoint={args.checkpoint}")

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

    # --- 构建测试集 ---
    total = config.train_size + config.val_size + config.test_size
    data = build_synthetic_data(total=total, seed=config.seed)
    test_data = data[config.train_size + config.val_size:]
    from torch.utils.data import DataLoader
    test_ds = TextDataset(test_data, vocab, config.max_length)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # --- 加载模型 ---
    model = SimpleTextClassifier(
        vocab_size=actual_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"加载 checkpoint: {ckpt_path}")

    # --- 评估 ---
    label_names = ["negative", "positive"]
    accuracy, confusion, per_class, wrong = evaluate_full(
        model, test_loader, device, config.num_classes, label_names
    )

    # --- 打印结果 ---
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n各类别指标:")
    for c in per_class:
        print(f"  {c['class']:12s}  precision={c['precision']:.4f}  "
              f"recall={c['recall']:.4f}  f1={c['f1']:.4f}")
    print("\n混淆矩阵 (行=真实, 列=预测):")
    header = "         " + "  ".join(f"{n:>10}" for n in label_names)
    print(header)
    for i, row in enumerate(confusion):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"  {label_names[i]:8s} {row_str}")

    if wrong:
        print("\n错误样本示例（最多5条）:")
        for w in wrong:
            true_name = label_names[w["true"]] if label_names else str(w["true"])
            pred_name = label_names[w["pred"]] if label_names else str(w["pred"])
            text, _ = test_data[w["sample_offset"]]
            print(f"  [{w['sample_offset']}] 真实={true_name}  预测={pred_name}  文本={text!r}")

    # --- 保存 ---
    metrics = {
        "checkpoint": args.checkpoint,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    out_path = config.output_path / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存: {out_path}")


if __name__ == "__main__":
    main()
