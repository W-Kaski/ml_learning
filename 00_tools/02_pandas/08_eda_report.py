"""
02_pandas / 08_eda_report.py
==============================
Topic: Exploratory Data Analysis (EDA) workflow using pandas.

Covers:
  1. Dataset overview: shape, dtypes, head, info
  2. Univariate stats: describe, skew, kurtosis, mode
  3. Missing value audit
  4. Outlier audit (IQR method across all numeric columns)
  5. Correlation matrix and top correlated pairs
  6. Categorical column analysis (cardinality, top frequencies)
  7. Target variable analysis (class balance, feature–target corr)
  8. Generating a text EDA summary report
"""

import numpy as np
import pandas as pd

np.random.seed(99)

# ──────────────────────────────────────────────────────────────
# Build a realistic dataset (loan applications)
# ──────────────────────────────────────────────────────────────
n = 600

df = pd.DataFrame({
    "loan_amount":  np.random.lognormal(mean=10.0, sigma=0.8, size=n).round(0).clip(1000, 500000),
    "annual_income":np.random.lognormal(mean=11.0, sigma=0.6, size=n).round(0).clip(10000, 1000000),
    "credit_score": np.random.normal(680, 80, n).clip(300, 850).round(0),
    "dti_ratio":    np.random.beta(2, 5, n).round(4),    # debt-to-income
    "age":          np.random.randint(21, 65, n),
    "years_emp":    np.random.randint(0, 30, n),
    "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n, p=[0.40, 0.20, 0.40]),
    "loan_purpose":   np.random.choice(["debt_consolidation", "home_improvement",
                                        "car", "medical", "vacation"], n,
                                       p=[0.45, 0.20, 0.15, 0.12, 0.08]),
    "grade":          np.random.choice(["A", "B", "C", "D", "E"], n, p=[0.20, 0.30, 0.25, 0.15, 0.10]),
})

# Inject missing values
for col, rate in [("annual_income", 0.05), ("credit_score", 0.08), ("years_emp", 0.06)]:
    idx = np.random.choice(n, int(n * rate), replace=False)
    df.loc[idx, col] = np.nan

# Target: default (1 = defaulted)
log_odds = (
    -3
    + 0.3 * (df["dti_ratio"].fillna(df["dti_ratio"].median()))
    - 0.002 * (df["credit_score"].fillna(680))
    + 0.000005 * (df["loan_amount"])
)
prob = 1 / (1 + np.exp(-log_odds))
df["default"] = (np.random.rand(n) < prob).astype(int)

# ──────────────────────────────────────────────────────────────
# 1. Dataset overview
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"Shape : {df.shape}")
print(f"Rows  : {df.shape[0]}")
print(f"Cols  : {df.shape[1]}")
print(f"\nDtypes:")
print(df.dtypes.to_string())
print(f"\nFirst 5 rows:")
print(df.head(5).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 2. Univariate stats
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. UNIVARIATE STATS (numerics)")
print("=" * 60)

num_cols = df.select_dtypes(include="number").columns.tolist()
desc = df[num_cols].describe().T.round(2)
desc["skew"]     = df[num_cols].skew().round(3)
desc["kurtosis"] = df[num_cols].kurtosis().round(3)
print(desc[["mean", "std", "min", "50%", "max", "skew", "kurtosis"]].to_string())

# ──────────────────────────────────────────────────────────────
# 3. Missing value audit
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MISSING VALUE AUDIT")
print("=" * 60)

missing = pd.DataFrame({
    "count": df.isna().sum(),
    "pct":   (df.isna().mean() * 100).round(2),
}).query("count > 0").sort_values("pct", ascending=False)

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing.to_string())

rows_complete = df.dropna().shape[0]
print(f"\nComplete rows: {rows_complete} / {n}  ({rows_complete/n*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# 4. Outlier audit (IQR method)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. OUTLIER AUDIT (IQR, k=1.5)")
print("=" * 60)

outlier_report = []
for col in num_cols:
    s = df[col].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((s < lo) | (s > hi)).sum()
    if n_out > 0:
        outlier_report.append({
            "column": col,
            "n_outliers": n_out,
            "pct": round(n_out / len(s) * 100, 2),
            "lower_fence": round(lo, 2),
            "upper_fence": round(hi, 2),
        })

out_df = pd.DataFrame(outlier_report).sort_values("pct", ascending=False)
print(out_df.to_string(index=False) if len(out_df) else "No outliers detected.")

# ──────────────────────────────────────────────────────────────
# 5. Correlation matrix & top pairs
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CORRELATION MATRIX & TOP PAIRS")
print("=" * 60)

# Fill NAs for correlation computation
df_num = df[num_cols].fillna(df[num_cols].median())
corr = df_num.corr().round(3)

print("Correlation matrix:")
print(corr.to_string())

# Extract upper triangle pairs
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
corr_pairs["abs_corr"] = corr_pairs["correlation"].abs()
top10 = corr_pairs.sort_values("abs_corr", ascending=False).head(10)
print("\nTop 10 correlated pairs:")
print(top10.drop(columns="abs_corr").to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 6. Categorical analysis
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. CATEGORICAL COLUMN ANALYSIS")
print("=" * 60)

cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    vc = df[col].value_counts()
    print(f"\n{col}  (cardinality={df[col].nunique()})")
    print(vc.to_string())

# ──────────────────────────────────────────────────────────────
# 7. Target variable analysis
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TARGET VARIABLE ANALYSIS (default)")
print("=" * 60)

target = "default"
print(f"Class distribution:")
vc = df[target].value_counts()
pct = (df[target].value_counts(normalize=True) * 100).round(2)
print(pd.concat([vc, pct.rename("pct%")], axis=1).to_string())

# Mean of numeric features by target class
print("\nNumeric feature means by default class:")
means = df_num.groupby(df[target]).mean().round(2).T
means.columns = ["No Default (0)", "Default (1)"]
means["diff%"] = (
    (means["Default (1)"] - means["No Default (0)"]) / means["No Default (0)"] * 100
).round(2)
print(means.to_string())

# Default rate by categorical features
print("\nDefault rate by grade:")
print(df.groupby("grade")[target].mean().round(4).sort_values(ascending=False).to_string())

print("\nDefault rate by loan_purpose:")
print(df.groupby("loan_purpose")[target].mean().round(4).sort_values(ascending=False).to_string())

# ──────────────────────────────────────────────────────────────
# 8. Text EDA summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. AUTO-GENERATED EDA SUMMARY REPORT")
print("=" * 60)

report_lines = [
    f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns",
    f"Numeric features  : {len(num_cols)} → {num_cols}",
    f"Categorical features: {len(cat_cols)} → {cat_cols}",
    "",
    "--- Missing Values ---",
]
if len(missing):
    for _, row in missing.iterrows():
        report_lines.append(f"  {row.name}: {int(row['count'])} missing ({row['pct']}%)")
else:
    report_lines.append("  No missing values.")

report_lines += [
    "",
    "--- Outliers (IQR) ---",
]
for _, row in out_df.iterrows():
    report_lines.append(f"  {row['column']}: {int(row['n_outliers'])} outliers ({row['pct']}%)")

target_rate = df[target].mean() * 100
report_lines += [
    "",
    f"--- Target ({target}) ---",
    f"  Default rate: {target_rate:.2f}%",
    f"  Class balance: {'Balanced' if 40 < target_rate < 60 else 'Imbalanced'}",
]

top_pair = top10.iloc[0]
report_lines += [
    "",
    "--- Top Correlations ---",
    f"  Highest: {top_pair['feature_1']} ↔ {top_pair['feature_2']} = {top_pair['correlation']:.3f}",
]

print("\n".join(report_lines))
print("\n[Done] 08_eda_report.py completed successfully.")
