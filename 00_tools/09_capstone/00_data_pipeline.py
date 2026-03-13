"""
09_capstone / 00_data_pipeline.py
信用风险分类项目：数据生成、清洗、特征工程。
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "credit_risk_raw.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "credit_risk_processed.csv")
PROFILE_PATH = os.path.join(DATA_DIR, "data_profile.json")


def build_raw_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        weights=[0.78, 0.22],
        class_sep=1.1,
        random_state=seed,
    )

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    # Map synthetic features to business-style fields
    df["age"] = (35 + 12 * df["f0"]).round().astype(int)
    df["income"] = (85000 + 26000 * df["f1"] + rng.normal(0, 8000, n)).round(0)
    df["loan_amount"] = (30000 + 18000 * df["f2"] + rng.normal(0, 6000, n)).round(0)
    df["credit_utilization"] = np.clip(0.45 + 0.2 * df["f3"] + rng.normal(0, 0.08, n), 0, 1)
    df["late_payments_12m"] = np.clip((2.0 + 2.8 * df["f4"] + rng.normal(0, 1.2, n)).round(), 0, 15)

    region_labels = np.array(["North", "South", "East", "West"])
    channel_labels = np.array(["Branch", "Online", "Partner"])
    df["region"] = region_labels[rng.integers(0, len(region_labels), n)]
    df["channel"] = channel_labels[rng.integers(0, len(channel_labels), n)]

    df["default"] = y.astype(int)

    # Inject missingness and mild outliers
    miss_idx_income = rng.choice(n, size=int(0.03 * n), replace=False)
    miss_idx_util = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[miss_idx_income, "income"] = np.nan
    df.loc[miss_idx_util, "credit_utilization"] = np.nan

    out_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[out_idx, "loan_amount"] *= 2.2

    keep_cols = [
        "age",
        "income",
        "loan_amount",
        "credit_utilization",
        "late_payments_12m",
        "region",
        "channel",
        "default",
    ]
    return df[keep_cols]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Robust clipping and imputation
    out["age"] = out["age"].clip(lower=18, upper=75)
    out["income"] = out["income"].clip(lower=15000, upper=300000)
    out["loan_amount"] = out["loan_amount"].clip(lower=1000, upper=250000)
    out["credit_utilization"] = out["credit_utilization"].clip(lower=0, upper=1)
    out["late_payments_12m"] = out["late_payments_12m"].clip(lower=0, upper=15)

    out["income"] = out["income"].fillna(out["income"].median())
    out["credit_utilization"] = out["credit_utilization"].fillna(out["credit_utilization"].median())

    # Feature engineering
    out["debt_to_income"] = out["loan_amount"] / (out["income"] + 1e-6)
    out["payment_stress"] = out["credit_utilization"] * (1 + out["late_payments_12m"] / 5)
    out["is_online"] = (out["channel"] == "Online").astype(int)

    return out


def main() -> None:
    raw_df = build_raw_data()
    raw_df.to_csv(RAW_PATH, index=False)

    clean_df = clean_and_engineer(raw_df)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    profile = {
        "raw_rows": int(raw_df.shape[0]),
        "processed_rows": int(clean_df.shape[0]),
        "target_rate": float(clean_df["default"].mean()),
        "null_counts": {k: int(v) for k, v in clean_df.isnull().sum().to_dict().items()},
        "columns": list(clean_df.columns),
    }
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print("raw saved:", RAW_PATH)
    print("processed saved:", PROCESSED_PATH)
    print("target rate:", round(profile["target_rate"], 4))
    print("[Done] 00_data_pipeline.py completed successfully.")


if __name__ == "__main__":
    main()
