#!/usr/bin/env python3

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(config):
    model_name = config.model_name.lower()
    if model_name in {"simplecnn", "simple_cnn"}:
        return SimpleCNN(num_classes=config.num_classes)

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {config.model_name}")