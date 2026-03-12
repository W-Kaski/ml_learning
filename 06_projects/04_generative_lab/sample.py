#!/usr/bin/env python3

import argparse
import json
import math
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_available_models(), default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    if args.model is not None:
        config.model_name = args.model
    num_samples = args.num_samples or config.sample_count

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config.checkpoint_path(config.model_name, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}\n请先运行 train.py --model {config.model_name}")

    bundle = build_model_bundle(config, config.model_name, device)
    load_bundle(bundle, checkpoint_path, device)
    images = sample_images(bundle, config, num_samples, device).cpu()
    image_path = config.sample_image_path(config.model_name, args.checkpoint)
    save_image(denormalize_images(images), image_path, nrow=max(1, int(math.sqrt(num_samples))))

    stats = {
        "model": config.model_name,
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "mean": round(images.mean().item(), 4),
        "std": round(images.std().item(), 4),
        "min": round(images.min().item(), 4),
        "max": round(images.max().item(), 4),
    }
    with open(config.sample_stats_path(config.model_name, args.checkpoint), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"=== 04_generative_lab 采样 ===")
    print(f"model={config.model_name}, checkpoint={args.checkpoint}, num_samples={num_samples}")
    print(f"sample image: {image_path}")
    print(f"sample stats: {config.sample_stats_path(config.model_name, args.checkpoint)}")


if __name__ == "__main__":
    main()