"""
08_mlflow / 05_stage_transition.py
Staging/Production 切换示例。
"""

import os
import mlflow
from mlflow import MlflowClient

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"
MODEL_NAME = "demo_logreg_model"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError("No model versions found. Run 04_registry_basics.py first.")

    latest = max(versions, key=lambda v: int(v.version))
    v = latest.version

    # Stage transition (legacy but still common in many teams)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=v,
        stage="Staging",
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=v,
        stage="Production",
        archive_existing_versions=True,
    )

    mv = client.get_model_version(name=MODEL_NAME, version=v)
    print("model:", MODEL_NAME, "version:", v, "current_stage:", mv.current_stage)

    # Optional modern alias style
    client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=v)
    champion = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")
    print("alias champion -> version", champion.version)
    print("[Done] 05_stage_transition.py completed successfully.")


if __name__ == "__main__":
    main()
