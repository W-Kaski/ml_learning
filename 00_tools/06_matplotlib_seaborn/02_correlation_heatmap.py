"""
06_matplotlib_seaborn / 02_correlation_heatmap.py
相关性热力图。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=130)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=False,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Breast Cancer Dataset)")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "02_correlation_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
    print("[Done] 02_correlation_heatmap.py completed successfully.")


if __name__ == "__main__":
    main()
