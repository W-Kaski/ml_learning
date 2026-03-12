#!/usr/bin/env python3

import argparse
import json

import torch

from config import Config
from dataset import build_dataloaders
from model import SimpleCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for image classification")
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--index", type=int, default=0, help="sample index in test set")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = config.best_ckpt_path if args.checkpoint == "best" else config.last_ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    _, _, test_loader = build_dataloaders(config)
    model = SimpleCNN(num_classes=config.num_classes).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)

    sample_index = args.index % images.size(0)
    image = images[sample_index : sample_index + 1].to(device)
    true_label = labels[sample_index].item()

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_label = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred_label].item())

    topk = torch.topk(probs, k=min(3, config.num_classes))
    top_predictions = [
        {"class": int(cls_idx.item()), "prob": float(prob.item())}
        for prob, cls_idx in zip(topk.values, topk.indices)
    ]

    infer_result = {
        "checkpoint": str(ckpt_path),
        "sample_index": sample_index,
        "true_label": true_label,
        "pred_label": pred_label,
        "pred_confidence": pred_conf,
        "top_predictions": top_predictions,
    }
    result_path = config.output_path / "infer_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(infer_result, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("01_image_classification/infer.py")
    print("=" * 70)
    print(f"checkpoint={ckpt_path}")
    print(f"sample_index={sample_index}, true_label={true_label}")
    print(f"pred_label={pred_label}, confidence={pred_conf:.2%}")
    print("top-3 predictions:")
    for item in top_predictions:
        print(f"  class={item['class']} prob={item['prob']:.2%}")
    print(f"result saved to {result_path}")


if __name__ == "__main__":
    main()