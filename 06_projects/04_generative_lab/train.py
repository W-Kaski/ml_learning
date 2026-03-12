#!/usr/bin/env python3

import argparse
import json
import random

import torch

from config import Config
from dataset import build_dataloaders
from models import build_model_bundle, evaluate_step, list_available_models, save_bundle, train_step


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_metrics(metric_list):
    keys = metric_list[0].keys()
    return {key: sum(item[key] for item in metric_list) / len(metric_list) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_available_models(), default=None)
    args = parser.parse_args()

    config = Config()
    if args.model is not None:
        config.model_name = args.model

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 04_generative_lab 训练 ===")
    print(f"model={config.model_name}, device={device}, epochs={config.epochs}, batch_size={config.batch_size}")

    train_loader, val_loader = build_dataloaders(config)
    bundle = build_model_bundle(config, config.model_name, device)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = []
        for images, _ in train_loader:
            train_metrics.append(train_step(bundle, images, config, device))

        val_metrics = []
        for images, _ in val_loader:
            val_metrics.append(evaluate_step(bundle, images, config, device))

        train_avg = aggregate_metrics(train_metrics)
        val_avg = aggregate_metrics(val_metrics)
        history_item = {"epoch": epoch}
        history_item.update({f"train_{k}": round(v, 4) for k, v in train_avg.items()})
        history_item.update({f"val_{k}": round(v, 4) for k, v in val_avg.items()})
        history.append(history_item)

        print(
            f"Epoch {epoch:02d}/{config.epochs}  "
            f"train_loss={train_avg['loss']:.4f}  val_loss={val_avg['loss']:.4f}"
        )

        if val_avg["loss"] < best_val_loss:
            best_val_loss = val_avg["loss"]
            save_bundle(bundle, config.checkpoint_path(config.model_name, "best"))
            print(f"  -> 保存 best checkpoint (val_loss={best_val_loss:.4f})")

    save_bundle(bundle, config.checkpoint_path(config.model_name, "last"))
    with open(config.history_path(config.model_name), "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n训练完成：{config.model_name}")
    print(f"best checkpoint: {config.checkpoint_path(config.model_name, 'best')}")
    print(f"last checkpoint: {config.checkpoint_path(config.model_name, 'last')}")
    print(f"history: {config.history_path(config.model_name)}")


if __name__ == "__main__":
    main()