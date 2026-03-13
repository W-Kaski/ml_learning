"""
04_sklearn / 06_metrics_classification.py
分类指标：Precision/Recall/F1/AUC。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("precision:", round(precision_score(y_test, pred), 4))
    print("recall   :", round(recall_score(y_test, pred), 4))
    print("f1       :", round(f1_score(y_test, pred), 4))
    print("roc_auc  :", round(roc_auc_score(y_test, proba), 4))
    print("confusion matrix:\n", confusion_matrix(y_test, pred))
    print("classification report:\n", classification_report(y_test, pred, digits=4))
    print("[Done] 06_metrics_classification.py completed successfully.")


if __name__ == "__main__":
    main()
