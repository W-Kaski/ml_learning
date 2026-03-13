"""
09_capstone / 02_eval_and_plots.py
评估指标与可视化报告输出。
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PRED_PATH = os.path.join(OUT_DIR, "test_predictions.csv")
PLOT_PATH = os.path.join(OUT_DIR, "evaluation_dashboard.png")
REPORT_PATH = os.path.join(OUT_DIR, "classification_report.txt")
SUMMARY_PATH = os.path.join(OUT_DIR, "eval_summary.json")


def main() -> None:
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError("Run 01_train_model.py first.")

    df = pd.read_csv(PRED_PATH)
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    y_proba = df["y_proba"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=130)

    axes[0, 0].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC={roc_auc:.4f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend(loc="lower right")

    axes[0, 1].plot(recall, precision, color="#2ca02c", linewidth=2, label=f"AUC={pr_auc:.4f}")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend(loc="lower left")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Pred")
    axes[1, 0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", cbar=False, ax=axes[1, 1])
    axes[1, 1].set_title("Confusion Matrix (Normalized)")
    axes[1, 1].set_xlabel("Pred")
    axes[1, 1].set_ylabel("True")

    fig.suptitle("Credit Risk Evaluation Dashboard", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(y_true, y_pred, digits=4)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({"roc_auc": roc_auc, "pr_auc": pr_auc}, f, ensure_ascii=False, indent=2)

    print("plot saved:", PLOT_PATH)
    print("report saved:", REPORT_PATH)
    print("[Done] 02_eval_and_plots.py completed successfully.")


if __name__ == "__main__":
    main()
