"""
08_mlflow / 04_registry_basics.py
模型注册基础。
"""

import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"
MODEL_NAME = "demo_logreg_model"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("08_mlflow_registry")

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    with mlflow.start_run(run_name="registry_source_run") as run:
        mlflow.log_param("model", "logreg")
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    reg = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)

    print("registered model:", MODEL_NAME)
    print("new version:", reg.version)
    print("top versions:", [v.version for v in versions_sorted[:5]])
    print("[Done] 04_registry_basics.py completed successfully.")


if __name__ == "__main__":
    main()
