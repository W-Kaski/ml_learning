"""
08_mlflow / 06_mlflow_with_xgboost.py
MLflow 与 XGBoost 集成。
"""

import os
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("08_mlflow_xgboost")

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    with mlflow.start_run(run_name="xgb_integration") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "auc": auc})
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")
        print("run_id:", run.info.run_id)

    print("accuracy:", round(acc, 4), "auc:", round(auc, 4))
    print("[Done] 06_mlflow_with_xgboost.py completed successfully.")


if __name__ == "__main__":
    main()
