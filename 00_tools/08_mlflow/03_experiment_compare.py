"""
08_mlflow / 03_experiment_compare.py
多实验运行并比较指标。
"""

import os
import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"


def run_once(n_estimators: int, max_depth: int | None) -> None:
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth if max_depth is not None else -1})
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
    })


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp_name = "08_mlflow_experiment_compare"
    mlflow.set_experiment(exp_name)

    settings = [(100, None), (200, 6), (300, 8), (400, None)]
    for n_est, depth in settings:
        with mlflow.start_run(run_name=f"rf_n{n_est}_d{depth}"):
            run_once(n_est, depth)

    exp = mlflow.get_experiment_by_name(exp_name)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"],
    )
    show_cols = [
        "run_id",
        "params.n_estimators",
        "params.max_depth",
        "metrics.accuracy",
        "metrics.f1_macro",
    ]
    print(runs[show_cols].head(5).to_string(index=False))
    print("[Done] 03_experiment_compare.py completed successfully.")


if __name__ == "__main__":
    main()
