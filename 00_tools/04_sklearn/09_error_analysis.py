"""
04_sklearn / 09_error_analysis.py
错误样本分析（分类）：查看被错分样本的特征分布。
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(classification_report(y_test, pred, digits=4))

    err_mask = pred != y_test.to_numpy()
    err_df = X_test.loc[err_mask].copy()
    err_df["y_true"] = y_test.loc[err_mask].to_numpy()
    err_df["y_pred"] = pred[err_mask]

    print("error count:", len(err_df), "/", len(X_test))

    if len(err_df) > 0:
        cols = [
            "mean radius",
            "mean texture",
            "mean smoothness",
            "mean concavity",
        ]
        summary = err_df[cols].describe().T[["mean", "std", "min", "max"]]
        print("\nerror sample stats:")
        print(summary)
        print("\nerror sample preview:")
        print(err_df[["y_true", "y_pred"] + cols].head(10))

    print("[Done] 09_error_analysis.py completed successfully.")


if __name__ == "__main__":
    main()
