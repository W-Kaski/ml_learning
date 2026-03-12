#!/usr/bin/env python3

from config import Config
from models import list_available_models


def main():
    config = Config()
    print("04_generative_lab/train.py")
    print(f"下一步实现统一训练入口，可选模型: {list_available_models()}, 当前={config.model_name}")


if __name__ == "__main__":
    main()