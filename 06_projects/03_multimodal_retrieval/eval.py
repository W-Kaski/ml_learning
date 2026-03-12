#!/usr/bin/env python3
"""
03_multimodal_retrieval/eval.py

在测试集上评估检索性能：
  - 图文对齐准确率（Top-1, Top-5）
  - Recall@K（文本检索图和图检索文本两个方向）
  - 平均精度均值 (mAP 简化版)
  - 结果保存至 outputs/eval_metrics.json

用法：
  python3 eval.py                       # 加载 best checkpoint
  python3 eval.py --checkpoint last
"""

import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import build_dataloaders, CLASS_NAMES
from model import DualEncoder


# ============================================================
# 第一节：导入所有图文嵌入
# ============================================================

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """返回 (img_embs, text_embs, labels) 全集在内存中的张量。"""
    img_embs, text_embs, labels_all = [], [], []
    model.eval()
    for imgs, texts, labels in loader:
        imgs, texts = imgs.to(device), texts.to(device)
        ie = model.encode_image(imgs).cpu()
        te = model.encode_text(texts).cpu()
        img_embs.append(ie)
        text_embs.append(te)
        labels_all.append(labels)
    return (
        torch.cat(img_embs),
        torch.cat(text_embs),
        torch.cat(labels_all),
    )


# ============================================================
# 第二节： Recall@K 计算
# ============================================================

def recall_at_k(query_embs, gallery_embs, query_labels, gallery_labels, k=5):
    """
    对每个 query，在 gallery 中索出最相似的 K 项，
    检查是否至少有一个与 query 类别相同。
    """
    # 余弦相似度矩阵 (N_q, N_g)
    sim = query_embs @ gallery_embs.T
    N = sim.size(0)
    hits = 0
    for i in range(N):
        top_k_idx = sim[i].topk(k).indices
        top_k_labels = gallery_labels[top_k_idx]
        if (top_k_labels == query_labels[i]).any():
            hits += 1
    return hits / N


# ============================================================
# 第三节：主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 03_multimodal_retrieval 评估 ===  checkpoint={args.checkpoint}")

    ckpt_path = config.best_ckpt_path if args.checkpoint == "best" else config.last_ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}\n请先运行 train.py")

    vocab_path = config.output_path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"词表不存在: {vocab_path}\n请先运行 train.py")
    with open(vocab_path) as f:
        token2idx = json.load(f)
    actual_vocab = len(token2idx)

    _, _, test_loader, _ = build_dataloaders(config)

    model = DualEncoder(
        image_channels=config.image_channels,
        image_feature_dim=config.image_feature_dim,
        text_vocab_size=actual_vocab,
        embed_dim=config.embed_dim,
        temperature=config.temperature,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"加载 checkpoint: {ckpt_path}")

    img_embs, text_embs, labels = extract_embeddings(model, test_loader, device)

    # 计算 Recall@K（两个方向）
    k = config.top_k
    r_i2t = recall_at_k(img_embs,  text_embs, labels, labels, k=k)
    r_t2i = recall_at_k(text_embs, img_embs,  labels, labels, k=k)
    r_i2t1 = recall_at_k(img_embs,  text_embs, labels, labels, k=1)
    r_t2i1 = recall_at_k(text_embs, img_embs,  labels, labels, k=1)

    print(f"\n图检索文（I2T）: Recall@1={r_i2t1:.4f}  Recall@{k}={r_i2t:.4f}")
    print(f"文检索图（T2I）: Recall@1={r_t2i1:.4f}  Recall@{k}={r_t2i:.4f}")
    print(f"均均 Recall@1={((r_i2t1+r_t2i1)/2):.4f}  Recall@{k}={((r_i2t+r_t2i)/2):.4f}")

    metrics = {
        "checkpoint": args.checkpoint,
        "num_test_samples": int(labels.size(0)),
        "I2T_recall_at_1": round(r_i2t1, 4),
        f"I2T_recall_at_{k}": round(r_i2t, 4),
        "T2I_recall_at_1": round(r_t2i1, 4),
        f"T2I_recall_at_{k}": round(r_t2i, 4),
        "mean_recall_at_1": round((r_i2t1 + r_t2i1) / 2, 4),
        f"mean_recall_at_{k}": round((r_i2t + r_t2i) / 2, 4),
    }
    out_path = config.output_path / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存: {out_path}")


if __name__ == "__main__":
    main()
