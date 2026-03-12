#!/usr/bin/env python3
"""
03_multimodal_retrieval/retrieve.py

单条检索演示：
  - 文本 → 图像：给定一段文本，返回最相似的 Top-K 图像
  - 图像 → 文本：给定一张图像，返回最相似的 Top-K 文本
  - 显示相似度分数和类别标签

用法：
  python3 retrieve.py --text "a photo of a cat" --topk 3
  python3 retrieve.py --image_index 5 --topk 3
  python3 retrieve.py --demo         # 同时展示两个方向
"""

import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import build_dataloaders, encode_text, CLASS_NAMES
from model import DualEncoder


# ============================================================
# 第一节：展示辅助
# ============================================================

def describe_image(label: int) -> str:
    name = CLASS_NAMES[label % len(CLASS_NAMES)]
    return f"[class={name}({label})]  合成图像"


# ============================================================
# 第二节：构建全库嵌入
# ============================================================

@torch.no_grad()
def build_gallery(model, loader, device):
    """返回 (img_embs, text_embs, labels, raw_texts)。"""
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
# 第三节：检索函数
# ============================================================

def text_to_image(query_text_emb, gallery_img_embs, gallery_labels, topk):
    sim = (query_text_emb @ gallery_img_embs.T).squeeze(0)  # (N,)
    scores, indices = sim.topk(topk)
    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        results.append({
            "rank": len(results) + 1,
            "similarity": round(score, 4),
            "label": int(gallery_labels[idx]),
            "class": CLASS_NAMES[int(gallery_labels[idx]) % len(CLASS_NAMES)],
            "description": describe_image(int(gallery_labels[idx])),
        })
    return results


def image_to_text(query_img_emb, gallery_text_embs, gallery_labels, topk):
    sim = (query_img_emb @ gallery_text_embs.T).squeeze(0)  # (N,)
    scores, indices = sim.topk(topk)
    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        results.append({
            "rank": len(results) + 1,
            "similarity": round(score, 4),
            "label": int(gallery_labels[idx]),
            "class": CLASS_NAMES[int(gallery_labels[idx]) % len(CLASS_NAMES)],
        })
    return results


# ============================================================
# 第四节：主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",        type=str,  default=None,
                        help="输入查询文本，与 --topk 配合进行文检索图")
    parser.add_argument("--image_index", type=int,  default=None,
                        help="从测试集取第 N 张图检索文本")
    parser.add_argument("--topk",        type=int,  default=3)
    parser.add_argument("--checkpoint",  choices=["best", "last"], default="best")
    parser.add_argument("--demo",        action="store_true",
                        help="同时展示文测图和图测文两个方向")
    args = parser.parse_args()

    if args.text is None and args.image_index is None and not args.demo:
        args.demo = True  # 默认全展示

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"=== 03_multimodal_retrieval 检索 ===  checkpoint={args.checkpoint}")

    # 构建全库嵌入
    img_embs, text_embs, labels = build_gallery(model, test_loader, device)
    print(f"库大小: {img_embs.size(0)} 条样本")

    retrieval_result = {}

    # ---- 文检图 ----
    if args.text or args.demo:
        query_text = args.text if args.text else "a photo of a cat"
        text_ids = encode_text(query_text, token2idx, config.text_max_length)
        text_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            q_emb = model.encode_text(text_tensor).cpu()
        results = text_to_image(q_emb, img_embs, labels, args.topk)
        print(f"\n[Text -> Image]  查询: {query_text!r}")
        for r in results:
            print(f"  Top-{r['rank']}: class={r['class']:8s}  similarity={r['similarity']:.4f}")
        retrieval_result["text_to_image"] = {"query": query_text, "results": results}

    # ---- 图检文 ----
    if args.image_index is not None or args.demo:
        idx = args.image_index if args.image_index is not None else 0
        if idx >= img_embs.size(0):
            raise IndexError(f"--image_index {idx} 超出库大小 {img_embs.size(0)}")
        q_emb = img_embs[idx:idx+1]
        q_label = int(labels[idx])
        results = image_to_text(q_emb, text_embs, labels, args.topk)
        print(f"\n[Image -> Text]  查询: 索引={idx}  真实类别={CLASS_NAMES[q_label % len(CLASS_NAMES)]}")
        for r in results:
            print(f"  Top-{r['rank']}: class={r['class']:8s}  similarity={r['similarity']:.4f}")
        retrieval_result["image_to_text"] = {
            "query_index": idx, "true_class": CLASS_NAMES[q_label % len(CLASS_NAMES)],
            "results": results,
        }

    out_path = config.output_path / "retrieve_result.json"
    with open(out_path, "w") as f:
        json.dump(retrieval_result, f, indent=2, ensure_ascii=False)
    print(f"\n检索结果已保存: {out_path}")


if __name__ == "__main__":
    main()

    main()