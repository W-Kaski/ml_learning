#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Config:
    image_size: int = 32
    latent_dim: int = 16
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5
    model_name: str = "vae"
    output_dir: str = "outputs"