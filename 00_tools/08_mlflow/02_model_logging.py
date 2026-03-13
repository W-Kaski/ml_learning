"""
08_mlflow / 02_model_logging.py
mlflow.sklearn 模型记录与加载。
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("08_mlflow_model_logging")

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    with mlflow.start_run(run_name="logreg_model") as run:
        mlflow.log_params({"model": "logreg", "max_iter": 3000})
        mlflow.log_metrics({"accuracy": acc, "auc": auc})
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        print("run_id:", run.info.run_id)
        print("model_uri:", model_uri)

    loaded_model = mlflow.sklearn.load_model(model_uri)
    pred2 = loaded_model.predict(X_test)
    print("reload consistency:", bool((pred2 == pred).all()))
    print("[Done] 02_model_logging.py completed successfully.")


if __name__ == "__main__":
    main()
