"""
02_pandas / 07_feature_engineering.py
========================================
Topic: Creating new features from raw data using pandas.

Covers:
  1. Binning / bucketing (pd.cut, pd.qcut)
  2. Encoding categoricals (one-hot, label, ordinal, target encoding)
  3. Interaction features (product, ratio, difference)
  4. Lag and window features for time-series
  5. Text-based features (str accessor: length, contains, extract)
  6. Date-derived features
  7. Feature scaling (min-max, standardization) via pandas
  8. Final feature summary
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# Build base dataset
# ──────────────────────────────────────────────────────────────
n = 500

df = pd.DataFrame({
    "age":       np.random.randint(18, 70, n),
    "income":    np.random.normal(60000, 20000, n).clip(15000, 200000).round(0),
    "score":     np.random.uniform(300, 850, n).round(0),
    "education": np.random.choice(
                     ["High School", "Bachelor", "Master", "PhD"],
                     n, p=[0.30, 0.40, 0.20, 0.10]),
    "city":      np.random.choice(["NYC", "LA", "Chicago", "Miami"], n,
                                  p=[0.35, 0.30, 0.20, 0.15]),
    "product":   np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
    "purchase":  np.random.choice([0, 1], n, p=[0.65, 0.35]),
    "joined":    pd.date_range("2020-01-01", periods=n, freq="D"),
    "email":     [f"user{i}@{'gmail' if i%3==0 else 'yahoo' if i%3==1 else 'work'}.com"
                  for i in range(n)],
    "note":      np.random.choice(
                     ["good customer", "new user", "VIP Gold", "churn risk", "new user"],
                     n),
})
print(f"Base dataset shape: {df.shape}")

# ──────────────────────────────────────────────────────────────
# 1. Binning
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. BINNING (pd.cut / pd.qcut)")
print("=" * 60)

# Fixed-width bins
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Youth", "Young Adult", "Middle Age", "Senior"],
)
print("age_group distribution:")
print(df["age_group"].value_counts().sort_index().to_string())

# Quantile-based bins (equal-frequency)
df["income_quartile"] = pd.qcut(df["income"], q=4,
                                labels=["Q1_Low", "Q2", "Q3", "Q4_High"])
print("\nincome_quartile distribution:")
print(df["income_quartile"].value_counts().sort_index().to_string())

# Score tier with custom labels
df["credit_tier"] = pd.cut(
    df["score"],
    bins=[299, 579, 669, 739, 799, 851],
    labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"],
)
print("\ncredit_tier distribution:")
print(df["credit_tier"].value_counts().sort_index().to_string())

# ──────────────────────────────────────────────────────────────
# 2. Encoding categoricals
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CATEGORICAL ENCODING")
print("=" * 60)

# One-hot encoding
ohe = pd.get_dummies(df["city"], prefix="city", dtype=int)
print(f"One-hot columns: {ohe.columns.tolist()}")
print(ohe.head(3).to_string())

# Label encoding (integer mapping)
city_labels = {c: i for i, c in enumerate(df["city"].unique())}
df["city_label"] = df["city"].map(city_labels)
print(f"\nLabel map: {city_labels}")

# Ordinal encoding (meaningful order)
edu_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
df["edu_ordinal"] = df["education"].map(edu_order)
print(f"\nEducation ordinal sample: {df[['education','edu_ordinal']].head(5).to_string(index=False)}")

# Target encoding (mean of target grouped by category)
target_enc = df.groupby("city")["purchase"].mean().round(4)
df["city_target_enc"] = df["city"].map(target_enc)
print(f"\nTarget encoding (purchase rate by city):")
print(target_enc.to_string())

# ──────────────────────────────────────────────────────────────
# 3. Interaction features
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. INTERACTION FEATURES")
print("=" * 60)

df["income_per_age"]   = (df["income"] / df["age"]).round(2)
df["score_income_ratio"] = (df["score"] / df["income"] * 1000).round(4)
df["age_edu_interaction"] = df["age"] * df["edu_ordinal"]
df["high_income_flag"]  = (df["income"] > df["income"].quantile(0.75)).astype(int)
df["score_age_product"] = (df["score"] * df["age"]).round(0)

print(df[["age", "income", "score", "edu_ordinal",
          "income_per_age", "score_income_ratio",
          "age_edu_interaction", "high_income_flag"]].head(5).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 4. Lag and window features (time-series style)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. LAG AND WINDOW FEATURES")
print("=" * 60)

ts_df = pd.DataFrame({
    "date":  pd.date_range("2024-01-01", periods=60, freq="D"),
    "sales": np.random.poisson(lam=100, size=60) + np.arange(60) * 0.5,
}).set_index("date")

ts_df["lag1"]    = ts_df["sales"].shift(1)
ts_df["lag7"]    = ts_df["sales"].shift(7)
ts_df["roll7_mean"] = ts_df["sales"].rolling(7).mean().round(2)
ts_df["roll7_std"]  = ts_df["sales"].rolling(7).std().round(2)
ts_df["pct_chg1"]   = ts_df["sales"].pct_change(1).round(4)

print(ts_df.dropna().head(6).to_string())

# ──────────────────────────────────────────────────────────────
# 5. Text-based features
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TEXT FEATURES (str accessor)")
print("=" * 60)

df["email_domain"]  = df["email"].str.split("@").str[1].str.split(".").str[0]
df["email_len"]     = df["email"].str.len()
df["is_vip"]        = df["note"].str.contains("VIP", case=False).astype(int)
df["is_churn_risk"] = df["note"].str.contains("churn", case=False).astype(int)
df["note_word_count"] = df["note"].str.split().str.len()

# Extract numeric part from "user123@..."  using regex
df["user_id_num"] = df["email"].str.extract(r"user(\d+)@").astype(float)

print(df[["email", "email_domain", "email_len", "is_vip", "is_churn_risk"]].head(8).to_string(index=False))
print(f"\nDomain distribution:")
print(df["email_domain"].value_counts().to_string())

# ──────────────────────────────────────────────────────────────
# 6. Date-derived features
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. DATE-DERIVED FEATURES")
print("=" * 60)

reference_date = pd.Timestamp("2024-01-01")
df["days_since_join"]  = (reference_date - df["joined"]).dt.days
df["join_year"]        = df["joined"].dt.year
df["join_month"]       = df["joined"].dt.month
df["join_quarter"]     = df["joined"].dt.quarter
df["join_dayofweek"]   = df["joined"].dt.dayofweek          # 0=Mon
df["join_is_weekend"]  = (df["join_dayofweek"] >= 5).astype(int)

print(df[["joined", "days_since_join", "join_year", "join_month",
          "join_quarter", "join_is_weekend"]].head(6).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 7. Feature scaling
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FEATURE SCALING (min-max, z-score)")
print("=" * 60)

numeric_cols = ["age", "income", "score"]

# Min-max normalization [0, 1]
df_scaled = df[numeric_cols].copy()
for col in numeric_cols:
    col_min, col_max = df_scaled[col].min(), df_scaled[col].max()
    df_scaled[f"{col}_minmax"] = ((df_scaled[col] - col_min) / (col_max - col_min)).round(4)

# Z-score standardization
for col in numeric_cols:
    df_scaled[f"{col}_zscore"] = ((df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()).round(4)

print("Before scaling (first 3 rows):")
print(df_scaled[numeric_cols].head(3).to_string(index=False))
print("\nAfter scaling (first 3 rows):")
print(df_scaled[[f"{c}_minmax" for c in numeric_cols]].head(3).to_string(index=False))
print("\nZ-score range check:")
for col in numeric_cols:
    z = df_scaled[f"{col}_zscore"]
    print(f"  {col}_zscore: mean={z.mean():.4f}, std={z.std():.4f}")

# ──────────────────────────────────────────────────────────────
# 8. Final feature summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. FEATURE SUMMARY")
print("=" * 60)

all_features = [
    "age", "income", "score",
    "age_group", "income_quartile", "credit_tier",       # binned
    "city_label", "edu_ordinal", "city_target_enc",       # encoded
    "income_per_age", "score_income_ratio",               # interaction
    "high_income_flag", "age_edu_interaction",
    "email_domain", "email_len", "is_vip", "is_churn_risk",  # text
    "days_since_join", "join_month", "join_is_weekend",   # date
]
feat_df = df[all_features]
print(f"Total features built: {len(all_features)}")
print(f"\ndtypes:")
print(feat_df.dtypes.to_string())
print(f"\nMissing values per feature:")
missing = feat_df.isna().sum()
print(missing[missing > 0].to_string() if missing.any() else "  None")

print("\n[Done] 07_feature_engineering.py completed successfully.")
