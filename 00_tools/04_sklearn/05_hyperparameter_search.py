"""
04_sklearn / 05_hyperparameter_search.py
GridSearchCV 与 RandomizedSearchCV 示例。
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC()),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"],
        },
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    random = RandomizedSearchCV(
        pipe,
        param_distributions={
            "svc__C": np.logspace(-2, 2, 30),
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"],
        },
        n_iter=10,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    random.fit(X_train, y_train)

    pred_grid = grid.best_estimator_.predict(X_test)
    pred_rand = random.best_estimator_.predict(X_test)

    print("Grid best params:", grid.best_params_)
    print("Grid cv score  :", round(grid.best_score_, 4))
    print("Grid test acc  :", round(accuracy_score(y_test, pred_grid), 4))

    print("Random best params:", random.best_params_)
    print("Random cv score   :", round(random.best_score_, 4))
    print("Random test acc   :", round(accuracy_score(y_test, pred_rand), 4))
    print("[Done] 05_hyperparameter_search.py completed successfully.")


if __name__ == "__main__":
    main()
