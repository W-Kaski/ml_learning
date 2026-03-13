"""
08_mlflow / 07_repro_report.py
复现实验报告：汇总实验关键 run 与指标到 markdown。
"""

import os
import mlflow
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlruns_data', 'mlflow.db')}"
REPORT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(REPORT_DIR, exist_ok=True)


def safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([None] * len(df))


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    experiments = mlflow.search_experiments()
    exp_rows = []
    for e in experiments:
        if e.name.startswith("08_mlflow"):
            exp_rows.append((e.experiment_id, e.name))

    report_lines = ["# MLflow Repro Report", "", "## Experiments", ""]

    for exp_id, exp_name in sorted(exp_rows, key=lambda x: x[1]):
        runs = mlflow.search_runs(experiment_ids=[exp_id], max_results=20)
        if runs.empty:
            report_lines.append(f"- {exp_name}: no runs")
            continue

        runs = runs.sort_values("start_time", ascending=False)
        report_lines.append(f"### {exp_name}")
        report_lines.append("")

        cols = [
            "run_id",
            "status",
            "metrics.accuracy",
            "metrics.auc",
            "metrics.f1_macro",
            "params.model",
            "params.n_estimators",
            "params.max_depth",
        ]
        view = pd.DataFrame({c: safe_get(runs, c) for c in cols})
        report_lines.append(view.head(5).to_markdown(index=False))
        report_lines.append("")

    report_path = os.path.join(REPORT_DIR, "repro_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # log this report as artifact in a dedicated run
    mlflow.set_experiment("08_mlflow_repro_report")
    with mlflow.start_run(run_name="repro_report"):
        mlflow.log_artifact(report_path, artifact_path="reports")

    print("report saved:", report_path)
    print("[Done] 07_repro_report.py completed successfully.")


if __name__ == "__main__":
    main()
