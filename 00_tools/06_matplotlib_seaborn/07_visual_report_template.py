"""
06_matplotlib_seaborn / 07_visual_report_template.py
可复用模型评估可视化模板。
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
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, f1_score

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def build_report_figure(y_true, proba, pred, title: str, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    cm = confusion_matrix(y_true, pred)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(13, 8), dpi=130)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(fpr, tpr, color="#1f77b4", linewidth=2)
    ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set_title("ROC Curve")

    ax2.plot(recall, precision, color="#2ca02c", linewidth=2)
    ax2.set_title("PR Curve")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax3)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Pred")
    ax3.set_ylabel("True")

    bins = np.linspace(0, 1, 21)
    ax4.hist(proba[y_true == 0], bins=bins, alpha=0.6, label="true=0")
    ax4.hist(proba[y_true == 1], bins=bins, alpha=0.6, label="true=1")
    ax4.set_title("Score Distribution")
    ax4.legend()

    auc_score = roc_auc_score(y_true, proba)
    f1 = f1_score(y_true, pred)
    fig.suptitle(f"{title} | AUC={auc_score:.4f} | F1={f1:.4f}", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = os.path.join(OUT_DIR, "07_visual_report_template.png")
    build_report_figure(
        y_true=y_test.to_numpy(),
        proba=proba,
        pred=pred,
        title="Reusable Model Evaluation Report",
        out_path=out,
    )

    print("saved:", out)
    print("[Done] 07_visual_report_template.py completed successfully.")


if __name__ == "__main__":
    main()
