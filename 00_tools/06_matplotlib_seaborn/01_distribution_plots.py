"""
06_matplotlib_seaborn / 01_distribution_plots.py
直方图/KDE/箱线图。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(42)
    n = 1200
    df = pd.DataFrame(
        {
            "score": np.concatenate([
                rng.normal(68, 8, n // 2),
                rng.normal(82, 6, n // 2),
            ]),
            "group": ["A"] * (n // 2) + ["B"] * (n // 2),
        }
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)

    sns.histplot(data=df, x="score", hue="group", bins=30, kde=False, ax=axes[0], alpha=0.5)
    axes[0].set_title("Histogram")

    sns.kdeplot(data=df, x="score", hue="group", fill=True, common_norm=False, alpha=0.3, ax=axes[1])
    axes[1].set_title("KDE")

    sns.boxplot(data=df, x="group", y="score", palette="Set2", ax=axes[2])
    axes[2].set_title("Box Plot")

    fig.suptitle("Distribution Plots", y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "01_distribution_plots.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
    print("[Done] 01_distribution_plots.py completed successfully.")


if __name__ == "__main__":
    main()
