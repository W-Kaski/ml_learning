"""
06_matplotlib_seaborn / 06_error_analysis_dashboard.py
错误分析图表集合：概率分布、错分置信度、特征对比、残差风格图。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(classification_report(y_test, pred, digits=4))

    df = X_test.copy()
    df["y_true"] = y_test.to_numpy()
    df["y_pred"] = pred
    df["proba"] = proba
    df["is_error"] = (df["y_true"] != df["y_pred"]).astype(int)
    df["conf"] = np.where(df["y_pred"] == 1, df["proba"], 1 - df["proba"])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=130)

    sns.histplot(data=df, x="proba", hue="y_true", bins=25, alpha=0.5, ax=axes[0, 0])
    axes[0, 0].set_title("Predicted Probability by True Label")

    sns.histplot(data=df[df["is_error"] == 1], x="conf", bins=20, color="#d62728", ax=axes[0, 1])
    axes[0, 1].set_title("Confidence of Misclassified Samples")

    sns.scatterplot(
        data=df,
        x="mean radius",
        y="mean texture",
        hue="is_error",
        palette={0: "#1f77b4", 1: "#d62728"},
        alpha=0.75,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Feature Space with Error Highlight")

    err_rate_by_bin = (
        df.assign(proba_bin=pd.cut(df["proba"], bins=np.linspace(0, 1, 11), include_lowest=True))
          .groupby("proba_bin", observed=False)["is_error"]
          .mean()
          .reset_index()
    )
    axes[1, 1].plot(err_rate_by_bin.index, err_rate_by_bin["is_error"], marker="o", color="#9467bd")
    axes[1, 1].set_title("Error Rate by Probability Bin")
    axes[1, 1].set_xlabel("Probability Bin Index")
    axes[1, 1].set_ylabel("Error Rate")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "06_error_analysis_dashboard.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
    print("[Done] 06_error_analysis_dashboard.py completed successfully.")


if __name__ == "__main__":
    main()
