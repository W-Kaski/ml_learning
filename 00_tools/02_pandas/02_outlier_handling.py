"""
02_pandas / 02_outlier_handling.py
====================================
Topic: Detecting and handling outliers in pandas DataFrames.

Covers:
  1. Generating a dataset with realistic outliers
  2. Detection – Z-score method
  3. Detection – IQR (Tukey) method
  4. Detection – Modified Z-score (Hampel identifier)
  5. Capping / Winsorization (clip to percentile bounds)
  6. Log transformation to reduce skew
  7. Dropping outliers
  8. Comparison: original vs treated distributions
"""

import numpy as np
import pandas as pd

np.random.seed(0)

# ──────────────────────────────────────────────────────────────
# 1. Generate dataset with injected outliers
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATASET WITH INJECTED OUTLIERS")
print("=" * 60)

n = 300
salaries = np.random.normal(60000, 10000, n)
# Inject high-end outliers
salaries[np.random.choice(n, 8, replace=False)] = np.random.uniform(200000, 500000, 8)
# Inject low-end outliers
salaries[np.random.choice(n, 5, replace=False)] = np.random.uniform(-50000, -5000, 5)

ages = np.random.randint(20, 60, n).astype(float)
ages[np.random.choice(n, 4, replace=False)] = [150, 200, -5, 999]

df = pd.DataFrame({"salary": salaries, "age": ages})
print(df.describe().round(2).to_string())
print(f"\nRows: {len(df)}")

# ──────────────────────────────────────────────────────────────
# 2. Z-score method
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Z-SCORE METHOD  (|z| > 3)")
print("=" * 60)

def zscore_outlier_mask(series, threshold=3.0):
    """Return boolean mask where True = outlier."""
    z = (series - series.mean()) / series.std(ddof=1)
    return z.abs() > threshold

mask_z_salary = zscore_outlier_mask(df["salary"])
mask_z_age    = zscore_outlier_mask(df["age"])

print(f"salary outliers (z>3): {mask_z_salary.sum()}")
print(df.loc[mask_z_salary, "salary"].sort_values().to_string())
print(f"\nage outliers (z>3): {mask_z_age.sum()}")
print(df.loc[mask_z_age, "age"].sort_values().to_string())

# ──────────────────────────────────────────────────────────────
# 3. IQR method
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. IQR (TUKEY) METHOD  (< Q1-1.5·IQR or > Q3+1.5·IQR)")
print("=" * 60)

def iqr_outlier_mask(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    return (series < lower) | (series > upper), lower, upper

mask_iqr_salary, lo_s, hi_s = iqr_outlier_mask(df["salary"])
mask_iqr_age,    lo_a, hi_a = iqr_outlier_mask(df["age"])

print(f"salary IQR bounds : [{lo_s:.0f}, {hi_s:.0f}]")
print(f"salary outliers   : {mask_iqr_salary.sum()}")
print(f"\nage    IQR bounds : [{lo_a:.1f}, {hi_a:.1f}]")
print(f"age outliers      : {mask_iqr_age.sum()}")

# ──────────────────────────────────────────────────────────────
# 4. Modified Z-score (Hampel identifier) – robust to outliers
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. MODIFIED Z-SCORE (Hampel)  (|mz| > 3.5)")
print("=" * 60)

def modified_zscore_mask(series, threshold=3.5):
    """Uses median and MAD instead of mean/std → more robust."""
    median = series.median()
    MAD    = (series - median).abs().median()
    mz     = 0.6745 * (series - median) / (MAD + 1e-9)
    return mz.abs() > threshold

mask_mz_salary = modified_zscore_mask(df["salary"])
mask_mz_age    = modified_zscore_mask(df["age"])
print(f"salary Hampel outliers: {mask_mz_salary.sum()}")
print(f"age    Hampel outliers: {mask_mz_age.sum()}")

# ──────────────────────────────────────────────────────────────
# 5. Capping / Winsorization
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CAPPING / WINSORIZATION  (clip to [1%, 99%] percentile)")
print("=" * 60)

def winsorize(series, low=0.01, high=0.99):
    lo, hi = series.quantile(low), series.quantile(high)
    return series.clip(lower=lo, upper=hi), lo, hi

df_capped = df.copy()
df_capped["salary"], lo, hi = winsorize(df["salary"])
print(f"salary capped to [{lo:.0f}, {hi:.0f}]")
print(f"  Before: min={df['salary'].min():.0f}  max={df['salary'].max():.0f}")
print(f"  After : min={df_capped['salary'].min():.0f}  max={df_capped['salary'].max():.0f}")

df_capped["age"], lo_a, hi_a = winsorize(df["age"], low=0.005, high=0.995)
print(f"\nage capped to [{lo_a:.1f}, {hi_a:.1f}]")
print(f"  Before: min={df['age'].min():.0f}  max={df['age'].max():.0f}")
print(f"  After : min={df_capped['age'].min():.0f}  max={df_capped['age'].max():.0f}")

# ──────────────────────────────────────────────────────────────
# 6. Log transformation (reduce right skew)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. LOG TRANSFORMATION (reduce right skew)")
print("=" * 60)

# Only valid for positive values; shift if necessary
salary_pos = df["salary"].clip(lower=1)   # ensure positivity
salary_log = np.log1p(salary_pos)

skew_before = salary_pos.skew()
skew_after  = salary_log.skew()
print(f"Salary skewness  before log: {skew_before:.3f}")
print(f"Salary skewness  after  log: {skew_after:.3f}")
print(f"(closer to 0 = less skewed)")

# ──────────────────────────────────────────────────────────────
# 7. Dropping outliers
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. DROPPING OUTLIERS")
print("=" * 60)

# Use IQR mask to drop
combined_mask = mask_iqr_salary | mask_iqr_age
df_clean = df[~combined_mask].copy()
print(f"Rows before drop : {len(df)}")
print(f"Rows after  drop : {len(df_clean)}  (removed {combined_mask.sum()} outliers)")
print("\nClean salary stats:")
print(df_clean["salary"].describe().round(2).to_string())

# ──────────────────────────────────────────────────────────────
# 8. Comparison table
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. COMPARISON – salary stats under each strategy")
print("=" * 60)

comparison = pd.DataFrame({
    "strategy": ["original", "winsorized", "drop outliers"],
    "count":  [len(df),             len(df_capped),    len(df_clean)],
    "mean":   [df["salary"].mean(), df_capped["salary"].mean(), df_clean["salary"].mean()],
    "std":    [df["salary"].std(),  df_capped["salary"].std(),  df_clean["salary"].std()],
    "min":    [df["salary"].min(),  df_capped["salary"].min(),  df_clean["salary"].min()],
    "max":    [df["salary"].max(),  df_capped["salary"].max(),  df_clean["salary"].max()],
    "skew":   [df["salary"].skew(), df_capped["salary"].skew(), df_clean["salary"].skew()],
}).round(2)

print(comparison.to_string(index=False))

print("\n[Done] 02_outlier_handling.py completed successfully.")
