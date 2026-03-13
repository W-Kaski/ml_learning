"""
06_matplotlib_seaborn / 00_plot_basics.py
figure/axes/样式管理基础。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.linspace(0, 2 * np.pi, 300)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=130)

    axes[0].plot(x, y1, label="sin(x)", color="#1f77b4", linewidth=2)
    axes[0].plot(x, y2, label="cos(x)", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Line Plot")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("value")
    axes[0].legend()

    rng = np.random.default_rng(42)
    xs = rng.normal(0, 1, 150)
    ys = 0.6 * xs + rng.normal(0, 0.7, 150)
    axes[1].scatter(xs, ys, alpha=0.7, c="#2ca02c", edgecolor="white", linewidth=0.5)
    axes[1].set_title("Scatter Plot")
    axes[1].set_xlabel("feature_1")
    axes[1].set_ylabel("feature_2")

    fig.suptitle("Matplotlib Basics: Figure + Axes + Style", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "00_plot_basics.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
    print("[Done] 00_plot_basics.py completed successfully.")


if __name__ == "__main__":
    main()
