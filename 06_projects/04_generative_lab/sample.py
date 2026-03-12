#!/usr/bin/env python3

from config import Config


def main():
    config = Config()
    print("04_generative_lab/sample.py")
    print(f"下一步实现采样与保存，latent_dim={config.latent_dim}")


if __name__ == "__main__":
    main()