"""
04_sklearn / 08_model_comparison.py
多模型交叉验证对比（分类）。
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
    data = load_wine(as_frame=True)
    X, y = data.data, data.target

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    }

    result = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        result.append((name, scores.mean(), scores.std()))

    result.sort(key=lambda x: x[1], reverse=True)
    print("model comparison (accuracy cv mean ± std):")
    for name, mean, std in result:
        print(f"  {name:<8}: {mean:.4f} ± {std:.4f}")

    print("winner:", result[0][0])
    print("[Done] 08_model_comparison.py completed successfully.")


if __name__ == "__main__":
    main()
