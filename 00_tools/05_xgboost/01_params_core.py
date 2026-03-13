"""
05_xgboost / 01_params_core.py
核心参数影响：max_depth, learning_rate(eta), subsample。
"""

from itertools import product
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    grid = {
        "max_depth": [2, 4, 6],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.7, 1.0],
    }

    rows = []
    for depth, lr, subs in product(grid["max_depth"], grid["learning_rate"], grid["subsample"]):
        model = XGBClassifier(
            n_estimators=250,
            max_depth=depth,
            learning_rate=lr,
            subsample=subs,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "max_depth": depth,
            "learning_rate": lr,
            "subsample": subs,
            "acc": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
        })

    rows.sort(key=lambda r: r["auc"], reverse=True)

    print("Top parameter settings by AUC:")
    for r in rows[:5]:
        print(
            f"depth={r['max_depth']} lr={r['learning_rate']} subs={r['subsample']} "
            f"acc={r['acc']:.4f} auc={r['auc']:.4f}"
        )

    print("\nObservation:")
    print("- 深度过高通常更容易过拟合。")
    print("- learning_rate 小时常需更大 n_estimators。")
    print("- subsample < 1 常能提升泛化，代价是更高方差。")
    print("[Done] 01_params_core.py completed successfully.")


if __name__ == "__main__":
    main()
