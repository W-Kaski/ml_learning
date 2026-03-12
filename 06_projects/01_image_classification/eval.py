#!/usr/bin/env python3

import argparse
import json

import torch
import torch.nn as nn

from config import Config
from dataset import build_dataloaders, get_label_names
from model import build_model


def evaluate_with_confusion(model, loader, device, num_classes):
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                confusion[true_label.long(), pred_label.long()] += 1

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), confusion


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image classification model")
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = Config()
    ckpt_path = base_config.best_ckpt_path if args.checkpoint == "best" else base_config.last_ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = Config.from_dict(checkpoint.get("config", {}))
    _, _, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, confusion = evaluate_with_confusion(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=config.num_classes,
    )

    metrics = {
        "checkpoint": str(ckpt_path),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": confusion.tolist(),
    }
    metrics_path = config.output_path / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("01_image_classification/eval.py")
    print("=" * 70)
    print(f"checkpoint={ckpt_path}")
    print(f"model={config.model_name}, use_real_data={config.use_real_data}")
    print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.2%}")
    print(f"metrics saved to {metrics_path}")
    print("confusion matrix:")
    label_names = get_label_names(config)
    print("labels:", label_names)
    for row in confusion.tolist():
        print(row)


if __name__ == "__main__":
    main()