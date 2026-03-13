"""
06_matplotlib_seaborn / 03_time_series_plots.py
时间序列可视化。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")

    trend = np.linspace(100, 140, len(dates))
    seasonal = 8 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
    noise = rng.normal(0, 2, len(dates))
    value = trend + seasonal + noise

    df = pd.DataFrame({"date": dates, "value": value})
    df["rolling_7d"] = df["value"].rolling(7, min_periods=1).mean()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
    sns.lineplot(data=df, x="date", y="value", linewidth=1.2, alpha=0.6, label="daily", ax=ax)
    sns.lineplot(data=df, x="date", y="rolling_7d", linewidth=2.2, label="rolling_7d", ax=ax)

    ax.set_title("Time Series with Rolling Mean")
    ax.set_xlabel("date")
    ax.set_ylabel("metric")
    fig.autofmt_xdate()

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "03_time_series_plots.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
    print("[Done] 03_time_series_plots.py completed successfully.")


if __name__ == "__main__":
    main()
