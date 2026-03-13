"""
04_sklearn / 03_feature_selection.py
特征筛选：SelectKBest + RFE。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = LogisticRegression(max_iter=3000)

    pipe_kbest = Pipeline([
        ("kbest", SelectKBest(score_func=f_classif, k=10)),
        ("model", LogisticRegression(max_iter=3000)),
    ])

    pipe_rfe = Pipeline([
        ("rfe", RFE(estimator=base, n_features_to_select=10, step=1)),
        ("model", LogisticRegression(max_iter=3000)),
    ])

    base.fit(X_train, y_train)
    pipe_kbest.fit(X_train, y_train)
    pipe_rfe.fit(X_train, y_train)

    acc_base = accuracy_score(y_test, base.predict(X_test))
    acc_kbest = accuracy_score(y_test, pipe_kbest.predict(X_test))
    acc_rfe = accuracy_score(y_test, pipe_rfe.predict(X_test))

    print(f"Baseline accuracy   : {acc_base:.4f}")
    print(f"SelectKBest accuracy: {acc_kbest:.4f}")
    print(f"RFE accuracy        : {acc_rfe:.4f}")
    print("[Done] 03_feature_selection.py completed successfully.")


if __name__ == "__main__":
    main()
