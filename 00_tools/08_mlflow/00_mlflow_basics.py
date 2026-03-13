"""
08_mlflow / 00_mlflow_basics.py
run / params / metrics 基础记录。
"""

import os
import mlflow

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("08_mlflow_basics")

    with mlflow.start_run(run_name="baseline_run") as run:
        params = {"model": "logreg", "max_iter": 3000, "seed": 42}
        metrics = {"accuracy": 0.9632, "f1": 0.9715, "auc": 0.9881}

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("module", "08_mlflow")
        mlflow.set_tag("stage", "baseline")

        print("run_id:", run.info.run_id)
        print("params:", params)
        print("metrics:", metrics)

    print("tracking_uri:", TRACKING_URI)
    print("[Done] 00_mlflow_basics.py completed successfully.")


if __name__ == "__main__":
    main()
