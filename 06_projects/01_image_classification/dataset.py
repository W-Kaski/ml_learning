#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(image_size=32, is_train=False, use_real_data=False):
    transform_list = []
    if is_train and use_real_data:
        transform_list.extend(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return transforms.Compose(
        transform_list
    )


def build_fake_dataset(size=256, image_size=32, num_classes=10):
    transform = build_transforms(image_size=image_size, is_train=True, use_real_data=False)
    return datasets.FakeData(
        size=size,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=transform,
    )


def build_cifar10_datasets(config):
    train_transform = build_transforms(image_size=config.image_size, is_train=True, use_real_data=True)
    eval_transform = build_transforms(image_size=config.image_size, is_train=False, use_real_data=True)

    full_train = datasets.CIFAR10(root=config.data_root, train=True, download=True, transform=train_transform)
    eval_train = datasets.CIFAR10(root=config.data_root, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(root=config.data_root, train=False, download=True, transform=eval_transform)

    val_size = max(1, int(len(full_train) * config.val_ratio))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_indices, val_indices = random_split(range(len(full_train)), [train_size, val_size], generator=generator)

    train_subset = torch.utils.data.Subset(full_train, train_indices.indices)
    val_subset = torch.utils.data.Subset(eval_train, val_indices.indices)
    return train_subset, val_subset, test_set


def build_splits(train_size, val_size, test_size, image_size=32, num_classes=10, seed=42):
    total_size = train_size + val_size + test_size
    full_dataset = build_fake_dataset(size=total_size, image_size=image_size, num_classes=num_classes)
    generator = torch.Generator().manual_seed(seed)
    return random_split(full_dataset, [train_size, val_size, test_size], generator=generator)


def build_dataloaders(config):
    if config.use_real_data:
        train_set, val_set, test_set = build_cifar10_datasets(config)
    else:
        train_set, val_set, test_set = build_splits(
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            image_size=config.image_size,
            num_classes=config.num_classes,
            seed=config.seed,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def get_label_names(config):
    if config.use_real_data:
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return [str(idx) for idx in range(config.num_classes)]