"""
05_xgboost / 00_xgb_baseline.py
XGBoost 分类与回归基线。
"""

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import math


def classification_baseline() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print("[Classification]")
    print("accuracy:", round(accuracy_score(y_test, pred), 4))
    print("roc_auc :", round(roc_auc_score(y_test, proba), 4))


def regression_baseline() -> None:
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, pred)

    print("\n[Regression]")
    print("MAE :", round(mean_absolute_error(y_test, pred), 4))
    print("RMSE:", round(math.sqrt(mse), 4))
    print("R2  :", round(r2_score(y_test, pred), 4))


def main() -> None:
    classification_baseline()
    regression_baseline()
    print("\n[Done] 00_xgb_baseline.py completed successfully.")


if __name__ == "__main__":
    main()
