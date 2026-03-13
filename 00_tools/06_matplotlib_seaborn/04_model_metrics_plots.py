"""
06_matplotlib_seaborn / 04_model_metrics_plots.py
ROC/PR/阈值分析图。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000)),
    ])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds_pr = precision_recall_curve(y_test, prob)
    pr_auc = auc(recall, precision)

    thresholds = np.linspace(0.05, 0.95, 50)
    f1s = [f1_score(y_test, (prob >= t).astype(int)) for t in thresholds]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)

    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC={roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="#2ca02c", linewidth=2, label=f"AUC={pr_auc:.4f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    axes[2].plot(thresholds, f1s, color="#d62728", linewidth=2)
    axes[2].set_title("Threshold vs F1")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("F1")

    best_idx = int(np.argmax(f1s))
    axes[2].scatter([thresholds[best_idx]], [f1s[best_idx]], color="black", s=30)
    axes[2].annotate(
        f"best={thresholds[best_idx]:.2f}\nF1={f1s[best_idx]:.3f}",
        (thresholds[best_idx], f1s[best_idx]),
        xytext=(thresholds[best_idx] + 0.03, f1s[best_idx] - 0.05),
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    fig.suptitle("Model Metrics Plots", y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "04_model_metrics_plots.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
    print("[Done] 04_model_metrics_plots.py completed successfully.")


if __name__ == "__main__":
    main()
