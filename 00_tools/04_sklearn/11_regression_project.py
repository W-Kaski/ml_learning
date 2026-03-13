"""
04_sklearn / 11_regression_project.py
回归小项目：diabetes 回归 Pipeline + 随机搜索 + 测试评估。
"""

import math
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main() -> None:
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "rf__n_estimators": [200, 300, 500, 800],
            "rf__max_depth": [None, 4, 6, 8, 12],
            "rf__min_samples_split": [2, 4, 8, 12],
            "rf__min_samples_leaf": [1, 2, 4, 6],
            "rf__max_features": ["sqrt", "log2", None],
        },
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, pred)

    print("best params:", search.best_params_)
    print("best cv neg_mse:", round(search.best_score_, 4))
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    print("[Done] 11_regression_project.py completed successfully.")


if __name__ == "__main__":
    main()
