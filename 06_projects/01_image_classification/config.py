#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    image_size: int = 32
    num_classes: int = 10
    use_real_data: bool = False
    val_ratio: float = 0.1
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 8
    train_size: int = 512
    val_size: int = 128
    test_size: int = 128
    seed: int = 42
    model_name: str = "simple_cnn"
    pretrained: bool = False
    output_dir: str = "outputs"

    @property
    def project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def data_root(self) -> Path:
        return self.project_dir.parent.parent / "datasets"

    @property
    def output_path(self) -> Path:
        path = self.project_dir / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def best_ckpt_path(self) -> Path:
        return self.output_path / "best_model.pth"

    @property
    def last_ckpt_path(self) -> Path:
        return self.output_path / "last_model.pth"

    def to_dict(self):
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        return data

    @classmethod
    def from_dict(cls, data):
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)