#!/usr/bin/env python3

import json
import random

import torch
from torchvision.utils import save_image

from config import Config
from dataset import denormalize_images
from models import build_model_bundle, list_available_models, load_bundle, sample_images


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_diversity(images):
    flat = images.flatten(1)
    if flat.size(0) < 2:
        return 0.0
    distances = torch.cdist(flat, flat, p=2)
    mask = ~torch.eye(flat.size(0), dtype=torch.bool)
    return distances[mask].mean().item()


def main():
    config = Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    metrics = {}
    num_samples = min(8, config.sample_count)
    print("=== 04_generative_lab 对比 ===")

    for model_name in list_available_models():
        checkpoint_path = config.checkpoint_path(model_name, "best")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"缺少 checkpoint: {checkpoint_path}\n请先训练全部模型后再运行 compare.py")

        bundle = build_model_bundle(config, model_name, device)
        load_bundle(bundle, checkpoint_path, device)
        images = sample_images(bundle, config, num_samples, device).cpu()
        rows.append(denormalize_images(images))
        metrics[model_name] = {
            "mean": round(images.mean().item(), 4),
            "std": round(images.std().item(), 4),
            "diversity_l2": round(pairwise_diversity(images), 4),
        }
        print(
            f"{model_name:10s}  mean={metrics[model_name]['mean']:.4f}  "
            f"std={metrics[model_name]['std']:.4f}  diversity={metrics[model_name]['diversity_l2']:.4f}"
        )

    grid = torch.cat(rows, dim=0)
    save_image(grid, config.compare_grid_path, nrow=num_samples)
    with open(config.compare_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"compare grid: {config.compare_grid_path}")
    print(f"compare metrics: {config.compare_metrics_path}")


if __name__ == "__main__":
    main()