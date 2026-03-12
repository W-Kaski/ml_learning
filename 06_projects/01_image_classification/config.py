#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 8
    train_size: int = 512
    val_size: int = 128
    test_size: int = 128
    seed: int = 42
    model_name: str = "simple_cnn"
    output_dir: str = "outputs"

    @property
    def project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def output_path(self) -> Path:
        return self.project_dir / self.output_dir

    @property
    def best_ckpt_path(self) -> Path:
        return self.output_path / "best_model.pth"

    @property
    def last_ckpt_path(self) -> Path:
        return self.output_path / "last_model.pth"