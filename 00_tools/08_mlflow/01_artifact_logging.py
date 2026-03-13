"""
08_mlflow / 01_artifact_logging.py
图片/报告等 artifact 记录。
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("08_mlflow_artifacts")

    # Prepare local files
    report_path = os.path.join(ARTIFACT_DIR, "simple_report.json")
    fig_path = os.path.join(ARTIFACT_DIR, "simple_curve.png")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": "artifact logging demo", "metric": {"auc": 0.9812}}, f, ensure_ascii=False, indent=2)

    x = np.linspace(0, 1, 100)
    y = x ** 2
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(x, y)
    ax.set_title("Simple Curve")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    with mlflow.start_run(run_name="artifact_demo") as run:
        mlflow.log_param("artifact_type", "report+image")
        mlflow.log_artifact(report_path, artifact_path="reports")
        mlflow.log_artifact(fig_path, artifact_path="figures")
        print("run_id:", run.info.run_id)

    print("logged files:", report_path, fig_path)
    print("[Done] 01_artifact_logging.py completed successfully.")


if __name__ == "__main__":
    main()
