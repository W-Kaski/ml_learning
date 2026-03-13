"""
05_xgboost / 03_imbalance_handling.py
类别不平衡：比较默认模型与 scale_pos_weight。
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier


def train_and_report(name: str, model: XGBClassifier, X_train, X_test, y_train, y_test) -> None:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"\n[{name}]")
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba), 4))
    print(classification_report(y_test, pred, digits=4))


def main() -> None:
    X, y = make_classification(
        n_samples=12000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    spw = neg / max(pos, 1)

    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    weighted = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print(f"train class ratio: neg={neg}, pos={pos}, scale_pos_weight={spw:.2f}")
    train_and_report("baseline", base, X_train, X_test, y_train, y_test)
    train_and_report("scale_pos_weight", weighted, X_train, X_test, y_train, y_test)
    print("[Done] 03_imbalance_handling.py completed successfully.")


if __name__ == "__main__":
    main()
