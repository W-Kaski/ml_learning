"""
04_sklearn / 00_baseline_classification.py
逻辑回归二分类基线。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("dataset:", data.frame.shape)
    print(f"accuracy: {acc:.4f}")
    print("confusion matrix:\n", confusion_matrix(y_test, pred))
    print("classification report:\n", classification_report(y_test, pred, digits=4))
    print("[Done] 00_baseline_classification.py completed successfully.")


if __name__ == "__main__":
    main()
