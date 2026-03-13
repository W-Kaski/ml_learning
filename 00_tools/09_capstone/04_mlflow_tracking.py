"""
09_capstone / 04_mlflow_tracking.py
记录 capstone 训练实验与模型工件。
"""

import os
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_risk_processed.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"
REPORT_PATH = os.path.join(OUT_DIR, "mlflow_run_summary.md")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run 00_data_pipeline.py first.")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("09_capstone_credit_risk")

    df = pd.read_csv(DATA_PATH)
    y = df["default"].astype(int)
    X = pd.get_dummies(df.drop(columns=["default"]), columns=["region", "channel"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 400,
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

    with mlflow.start_run(run_name="capstone_xgb") as run:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "auc": float(roc_auc_score(y_test, proba)),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# Capstone MLflow Run Summary\n\n")
            f.write(f"- run_id: {run.info.run_id}\n")
            for k, v in metrics.items():
                f.write(f"- {k}: {v:.4f}\n")

        mlflow.log_artifact(REPORT_PATH, artifact_path="reports")

        print("run_id:", run.info.run_id)
        print("metrics:", {k: round(v, 4) for k, v in metrics.items()})

    print("tracking_uri:", TRACKING_URI)
    print("[Done] 04_mlflow_tracking.py completed successfully.")


if __name__ == "__main__":
    main()
