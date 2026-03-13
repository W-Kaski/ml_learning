"""
04_sklearn / 10_classification_project.py
分类小项目：wine 多分类 Pipeline + CV + 测试集评估。
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main() -> None:
    data = load_wine(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("train size:", len(X_train), "test size:", len(X_test))
    print("cv accuracy mean/std:", round(cv_scores.mean(), 4), round(cv_scores.std(), 4))
    print("test accuracy:", round(accuracy_score(y_test, pred), 4))
    print("confusion matrix:\n", confusion_matrix(y_test, pred))
    print("classification report:\n", classification_report(y_test, pred, digits=4))
    print("[Done] 10_classification_project.py completed successfully.")


if __name__ == "__main__":
    main()
