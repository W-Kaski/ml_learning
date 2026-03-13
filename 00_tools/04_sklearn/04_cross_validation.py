"""
04_sklearn / 04_cross_validation.py
KFold 与 StratifiedKFold 示例。
"""

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def main() -> None:
    data = load_wine(as_frame=True)
    X, y = data.data, data.target

    model = LogisticRegression(max_iter=4000)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    score_kf = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    score_skf = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    print("KFold scores       :", [round(x, 4) for x in score_kf])
    print("StratifiedKFold    :", [round(x, 4) for x in score_skf])
    print("KFold mean         :", round(score_kf.mean(), 4))
    print("StratifiedKFold mean:", round(score_skf.mean(), 4))
    print("[Done] 04_cross_validation.py completed successfully.")


if __name__ == "__main__":
    main()
