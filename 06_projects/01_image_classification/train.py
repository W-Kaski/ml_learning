#!/usr/bin/env python3

import json

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import build_dataloaders
from model import SimpleCNN


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def main():
    config = Config()
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.output_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(config)
    model = SimpleCNN(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print("=" * 70)
    print("01_image_classification/train.py")
    print("=" * 70)
    print(f"device={device}, epochs={config.epochs}, batch_size={config.batch_size}")
    print(f"output_dir={config.output_path}")
    print()

    history = []
    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch:2d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": config.__dict__,
                },
                config.best_ckpt_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": config.epochs,
            "val_acc": history[-1]["val_acc"] if history else 0.0,
            "config": config.__dict__,
        },
        config.last_ckpt_path,
    )

    history_path = config.output_path / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print()
    print(f"best checkpoint: {config.best_ckpt_path}")
    print(f"last checkpoint: {config.last_ckpt_path}")
    print(f"history: {history_path}")
    print(f"best val acc: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()