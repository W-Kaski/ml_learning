#!/usr/bin/env python3

from torchvision import datasets, transforms


def build_fake_image_dataset(size=256, image_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.FakeData(
        size=size,
        image_size=(3, image_size, image_size),
        num_classes=10,
        transform=transform,
    )