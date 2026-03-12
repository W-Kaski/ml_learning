#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_fake_image_dataset(size=320, image_size=32):
    return datasets.FakeData(
        size=size,
        image_size=(3, image_size, image_size),
        num_classes=10,
        transform=build_transforms(),
    )


def build_dataloaders(config):
    dataset = build_fake_image_dataset(size=config.dataset_size, image_size=config.image_size)
    val_size = max(1, int(len(dataset) * config.val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def denormalize_images(images):
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)