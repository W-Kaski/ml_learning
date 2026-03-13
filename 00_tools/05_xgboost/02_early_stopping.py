"""
05_xgboost / 02_early_stopping.py
提前停止：使用验证集寻找最佳迭代轮次。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    model = XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=60,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("best_iteration:", model.best_iteration)
    print("best_score    :", model.best_score)
    print("test accuracy :", round(accuracy_score(y_test, pred), 4))
    print("test auc      :", round(roc_auc_score(y_test, proba), 4))
    print("[Done] 02_early_stopping.py completed successfully.")


if __name__ == "__main__":
    main()
