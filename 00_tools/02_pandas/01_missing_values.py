"""
02_pandas / 01_missing_values.py
=================================
Topic: Detecting and handling missing values in pandas DataFrames.

Covers:
  1. Creating a dataset with realistic missing patterns
  2. Detecting missing values (isna / notna / info / isnull)
  3. Strategy A – Drop rows/columns (dropna)
  4. Strategy B – Fill with constant / median / mode
  5. Strategy C – Forward-fill / back-fill
  6. Strategy D – Group-wise fill (fill with group median)
  7. Strategy E – Interpolation (linear, polynomial)
  8. Comparing strategies side-by-side
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# 1. Build a messy dataset
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. BUILD DATASET WITH MISSING VALUES")
print("=" * 60)

n = 200
df_raw = pd.DataFrame({
    "age":       np.where(np.random.rand(n) < 0.10, np.nan,
                          np.random.randint(18, 65, n).astype(float)),
    "income":    np.where(np.random.rand(n) < 0.15, np.nan,
                          np.random.normal(50000, 15000, n).round(2)),
    "score":     np.where(np.random.rand(n) < 0.05, np.nan,
                          np.random.uniform(0, 100, n).round(1)),
    "category":  np.where(np.random.rand(n) < 0.08, np.nan,
                          np.random.choice(["A", "B", "C"], n)),
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
})
# Inject a column that is > 50% missing (useful to show threshold-based drop)
df_raw["sparse_col"] = np.where(np.random.rand(n) < 0.60, np.nan, 1.0)

print(df_raw.head(8).to_string())
print(f"\nShape: {df_raw.shape}")

# ──────────────────────────────────────────────────────────────
# 2. Detect missing values
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. MISSING VALUE DETECTION")
print("=" * 60)

missing_count = df_raw.isna().sum()
missing_pct   = (df_raw.isna().mean() * 100).round(2)
missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct":   missing_pct,
    "dtype":         df_raw.dtypes,
})
print(missing_report.to_string())

total_cells = df_raw.size
total_missing = df_raw.isna().sum().sum()
print(f"\nTotal cells : {total_cells}")
print(f"Total missing: {total_missing}  ({total_missing/total_cells*100:.2f}%)")

# Which rows have ANY missing value?
rows_with_na = df_raw.isna().any(axis=1).sum()
print(f"Rows with ≥1 NA : {rows_with_na} / {n}")

# ──────────────────────────────────────────────────────────────
# 3. Strategy A – dropna
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. STRATEGY A – dropna")
print("=" * 60)

# Drop rows where ALL values are NA (rarely useful, but safe baseline)
df_drop_all = df_raw.dropna(how="all")
print(f"dropna(how='all')  → {len(df_drop_all)} rows kept  (was {len(df_raw)})")

# Drop rows where ANY value is NA
df_drop_any = df_raw.dropna(how="any")
print(f"dropna(how='any')  → {len(df_drop_any)} rows kept")

# Drop columns above a missing threshold (e.g., >50% missing)
thresh_col = int(0.50 * n)                   # keep col if it has at least 50% valid
df_drop_col = df_raw.dropna(axis=1, thresh=thresh_col)
print(f"dropna(axis=1, thresh=50%) → {df_drop_col.shape[1]} cols kept  "
      f"(dropped: {set(df_raw.columns) - set(df_drop_col.columns)})")

# Drop rows with NA only in specific columns
df_drop_subset = df_raw.dropna(subset=["age", "income"])
print(f"dropna(subset=['age','income']) → {len(df_drop_subset)} rows kept")

# ──────────────────────────────────────────────────────────────
# 4. Strategy B – fill with constant / median / mode
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. STRATEGY B – fillna (constant / median / mode)")
print("=" * 60)

df_fill = df_raw.drop(columns=["sparse_col"]).copy()

# Constant fill
df_fill_const = df_fill.copy()
df_fill_const["age"]      = df_fill_const["age"].fillna(-1)         # sentinel
df_fill_const["category"] = df_fill_const["category"].fillna("Unknown")
print("Constant fill sample (age sentinels):",
      (df_fill_const["age"] == -1).sum(), "rows got -1")

# Median fill for numeric
df_fill_median = df_fill.copy()
for col in ["age", "income", "score"]:
    median_val = df_fill_median[col].median()
    df_fill_median[col] = df_fill_median[col].fillna(median_val)
    print(f"  {col}: filled NA with median={median_val:.2f}")

# Mode fill for categorical
df_fill_mode = df_fill.copy()
mode_val = df_fill_mode["category"].mode()[0]
df_fill_mode["category"] = df_fill_mode["category"].fillna(mode_val)
print(f"  category: filled NA with mode='{mode_val}'")

# Verify no more NAs in numeric cols
print("\nRemaining NAs after median fill:")
print(df_fill_median[["age", "income", "score"]].isna().sum().to_string())

# ──────────────────────────────────────────────────────────────
# 5. Strategy C – forward-fill / back-fill
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. STRATEGY C – ffill / bfill (time-series style)")
print("=" * 60)

# Simulate a time-series with gaps
ts = pd.Series(
    [1.0, np.nan, np.nan, 4.0, np.nan, 6.0, np.nan, np.nan, 9.0],
    name="sensor"
)
print("Original :", ts.tolist())
print("ffill    :", ts.ffill().tolist())
print("bfill    :", ts.bfill().tolist())

# limit= parameter: fill at most N consecutive NAs
print("ffill(limit=1):", ts.ffill(limit=1).tolist())

# Apply ffill to the full dataframe (sort by timestamp first)
df_ffill = df_fill.sort_values("timestamp").copy()
df_ffill[["age", "income", "score"]] = (
    df_ffill[["age", "income", "score"]].ffill()
)
remaining_after_ffill = df_ffill[["age", "income", "score"]].isna().sum().sum()
print(f"\nAfter ffill on full df, remaining NAs in numeric cols: {remaining_after_ffill}")

# ──────────────────────────────────────────────────────────────
# 6. Strategy D – group-wise fill
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. STRATEGY D – group-wise fill (fill with group median)")
print("=" * 60)

df_group = df_fill.copy()

# First fill category NAs so we can group on it
df_group["category"] = df_group["category"].fillna("Unknown")

# Fill income with the median income of the same category group
def fill_group_median(series):
    return series.fillna(series.median())

df_group["income"] = (
    df_group.groupby("category")["income"].transform(fill_group_median)
)

group_medians = df_group.groupby("category")["income"].median().round(2)
print("Group medians used for filling:")
print(group_medians.to_string())

still_missing = df_group["income"].isna().sum()
print(f"\nRemaining income NAs after group fill: {still_missing}")
# Any remaining gap belongs to groups that were entirely NA → fill with global median
df_group["income"] = df_group["income"].fillna(df_fill["income"].median())
print(f"After global fallback: {df_group['income'].isna().sum()} NAs left")

# ──────────────────────────────────────────────────────────────
# 7. Strategy E – interpolation
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. STRATEGY E – interpolation")
print("=" * 60)

s_with_gaps = pd.Series([0.0, np.nan, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0])
print("Original          :", s_with_gaps.tolist())
print("linear interp     :", s_with_gaps.interpolate(method="linear").tolist())
print("polynomial(deg=2) :", s_with_gaps.interpolate(method="polynomial", order=2).round(2).tolist())
print("nearest           :", s_with_gaps.interpolate(method="nearest").tolist())

# ──────────────────────────────────────────────────────────────
# 8. Side-by-side comparison
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. COMPARISON – mean of 'income' under each strategy")
print("=" * 60)

original_mean = df_raw["income"].mean()

# a. Drop approach
mean_drop = df_raw["income"].dropna().mean()

# b. Median fill
income_median_fill = df_raw["income"].fillna(df_raw["income"].median())
mean_median = income_median_fill.mean()

# c. ffill
income_ffill = df_raw["income"].ffill().bfill()   # bfill handles leading NAs
mean_ffill = income_ffill.mean()

# d. Interp
income_interp = df_raw["income"].interpolate()
mean_interp = income_interp.mean()

comparison = pd.DataFrame({
    "strategy": ["original (excl NA)", "drop", "median fill", "ffill+bfill", "interpolate"],
    "income_mean": [original_mean, mean_drop, mean_median, mean_ffill, mean_interp],
    "income_std" : [df_raw["income"].std(),
                    df_raw["income"].dropna().std(),
                    income_median_fill.std(),
                    income_ffill.std(),
                    income_interp.std()],
})
comparison["income_mean"] = comparison["income_mean"].round(2)
comparison["income_std"]  = comparison["income_std"].round(2)
print(comparison.to_string(index=False))

print("\n[Done] 01_missing_values.py completed successfully.")
