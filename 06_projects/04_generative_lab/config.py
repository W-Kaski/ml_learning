#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    image_size: int = 32
    image_channels: int = 3
    latent_dim: int = 32
    batch_size: int = 64
    learning_rate: float = 2e-4
    epochs: int = 4
    model_name: str = "vae"
    dataset_size: int = 320
    val_ratio: float = 0.2
    diffusion_steps: int = 20
    sample_count: int = 16
    num_workers: int = 0
    seed: int = 42
    output_dir: str = "outputs"

    @property
    def project_dir(self) -> Path:
        return Path(__file__).parent

    @property
    def output_path(self) -> Path:
        path = self.project_dir / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_output_dir(self, model_name: str | None = None) -> Path:
        name = model_name or self.model_name
        path = self.output_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def checkpoint_path(self, model_name: str | None = None, checkpoint_name: str = "best") -> Path:
        return self.model_output_dir(model_name) / f"{checkpoint_name}_model.pth"

    def history_path(self, model_name: str | None = None) -> Path:
        return self.model_output_dir(model_name) / "train_history.json"

    def sample_image_path(self, model_name: str | None = None, checkpoint_name: str = "best") -> Path:
        name = model_name or self.model_name
        return self.model_output_dir(name) / f"samples_{checkpoint_name}.png"

    def sample_stats_path(self, model_name: str | None = None, checkpoint_name: str = "best") -> Path:
        name = model_name or self.model_name
        return self.model_output_dir(name) / f"sample_stats_{checkpoint_name}.json"

    @property
    def compare_grid_path(self) -> Path:
        return self.output_path / "compare_grid.png"

    @property
    def compare_metrics_path(self) -> Path:
        return self.output_path / "compare_metrics.json"