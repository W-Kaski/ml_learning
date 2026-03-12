#!/usr/bin/env python3

from models import list_available_models


def main():
    print("04_generative_lab/compare.py")
    print(f"下一步实现不同生成模型的结果对比: {list_available_models()}")


if __name__ == "__main__":
    main()