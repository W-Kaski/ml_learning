"""
02_pandas / 05_pivot_reshape.py
================================
Topic: Reshaping DataFrames with pivot, melt, stack, unstack, and crosstab.

Covers:
  1. pivot_table – summary table with aggregation
  2. pivot (no aggregation, unique index required)
  3. melt – wide → long (unpivot)
  4. stack / unstack – MultiIndex reshaping
  5. pd.crosstab – frequency table
  6. Wide → long → wide round-trip
  7. Practical: reshape sales data for time-series analysis
"""

import numpy as np
import pandas as pd

np.random.seed(11)

# ──────────────────────────────────────────────────────────────
# 1. pivot_table
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. PIVOT TABLE")
print("=" * 60)

n = 300
df = pd.DataFrame({
    "region":  np.random.choice(["North", "South", "East", "West"], n),
    "product": np.random.choice(["A", "B", "C"], n),
    "quarter": np.random.choice(["Q1", "Q2", "Q3", "Q4"], n),
    "sales":   np.random.randint(100, 1000, n),
    "units":   np.random.randint(1, 50, n),
})

# Mean sales by region (rows) × product (columns)
pt = pd.pivot_table(
    df,
    values="sales",
    index="region",
    columns="product",
    aggfunc="mean",
    margins=True,          # add row/col totals
    margins_name="All",
).round(0)
print(pt.to_string())

# Multiple value columns
pt2 = pd.pivot_table(
    df,
    values=["sales", "units"],
    index="region",
    columns="quarter",
    aggfunc="sum",
    fill_value=0,
)
print("\nMulti-value pivot (first 4 columns):")
print(pt2.iloc[:, :4].to_string())

# ──────────────────────────────────────────────────────────────
# 2. pivot (no aggregation)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. WIDE FORMAT WITH pivot (unique index)")
print("=" * 60)

# Create a small tidy dataset with unique (city, year) combos
tidy = pd.DataFrame({
    "city":  ["NYC", "NYC", "LA", "LA", "Chi", "Chi"],
    "year":  [2022, 2023, 2022, 2023, 2022, 2023],
    "population_m": [8.3, 8.4, 4.0, 4.0, 2.7, 2.7],
})
wide = tidy.pivot(index="city", columns="year", values="population_m")
wide.columns.name = None   # clean up column name label
print(wide.to_string())

# ──────────────────────────────────────────────────────────────
# 3. melt – wide → long
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MELT – wide → long (unpivot)")
print("=" * 60)

wide_scores = pd.DataFrame({
    "student": ["Alice", "Bob", "Carol"],
    "math":    [90, 78, 85],
    "english": [88, 92, 79],
    "science": [76, 84, 91],
})
print("Wide format:")
print(wide_scores.to_string(index=False))

long_scores = wide_scores.melt(
    id_vars="student",
    value_vars=["math", "english", "science"],
    var_name="subject",
    value_name="score",
)
print("\nLong format (melted):")
print(long_scores.to_string(index=False))

# Aggregation is easier on long format
print("\nMean score per subject:")
print(long_scores.groupby("subject")["score"].mean().round(1).to_string())

# ──────────────────────────────────────────────────────────────
# 4. stack / unstack
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. STACK / UNSTACK")
print("=" * 60)

# Create a DataFrame with MultiIndex columns
arrays = [["Math", "Math", "Eng", "Eng"], ["midterm", "final", "midterm", "final"]]
midx   = pd.MultiIndex.from_arrays(arrays, names=["subject", "exam"])
df_grades = pd.DataFrame(
    np.random.randint(60, 100, (3, 4)),
    index=["Alice", "Bob", "Carol"],
    columns=midx,
)
print("Original (wide, MultiIndex cols):")
print(df_grades.to_string())

# stack: move innermost column level to row index
stacked = df_grades.stack(level="exam", future_stack=True)
print("\nAfter stack('exam'):")
print(stacked.to_string())

# unstack: inverse operation
unstacked = stacked.unstack("exam")
print("\nAfter unstack('exam') – back to original:")
print(unstacked.to_string())

# ──────────────────────────────────────────────────────────────
# 5. crosstab – frequency table
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CROSSTAB – frequency and normalized table")
print("=" * 60)

ct = pd.crosstab(df["region"], df["product"], margins=True, margins_name="Total")
print("Raw counts:")
print(ct.to_string())

ct_norm = pd.crosstab(df["region"], df["product"], normalize="index").round(3)
print("\nRow-normalized (proportions within region):")
print(ct_norm.to_string())

# With an aggregation value
ct_sales = pd.crosstab(
    df["region"], df["quarter"],
    values=df["sales"], aggfunc="sum"
).fillna(0).astype(int)
print("\nTotal sales by region × quarter:")
print(ct_sales.to_string())

# ──────────────────────────────────────────────────────────────
# 6. Wide → long → wide round-trip
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. WIDE → LONG → WIDE ROUND-TRIP")
print("=" * 60)

original_wide = wide_scores.copy()

# Step 1: melt
melted = original_wide.melt(id_vars="student", var_name="subject", value_name="score")

# Step 2: pivot back
reconstructed = melted.pivot(index="student", columns="subject", values="score")
reconstructed = reconstructed.reset_index()
reconstructed.columns.name = None

print("Round-trip matches original:", original_wide.equals(
    reconstructed[["student", "math", "english", "science"]]
))
print(reconstructed.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 7. Practical: monthly sales time-series per product
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. PRACTICAL – monthly sales trend per product")
print("=" * 60)

months = pd.date_range("2024-01", periods=12, freq="MS")
sales_long = pd.DataFrame([
    {"month": m, "product": p,
     "sales": np.random.randint(500, 2000)}
    for m in months for p in ["A", "B", "C"]
])

# Reshape to wide (product as columns) for easy comparison
sales_wide = sales_long.pivot(index="month", columns="product", values="sales")
sales_wide.columns.name = None
sales_wide.index = sales_wide.index.strftime("%b")   # friendly month names

print(sales_wide.to_string())
print(f"\nProduct totals:")
print(sales_wide.sum().sort_values(ascending=False).to_string())

print("\n[Done] 05_pivot_reshape.py completed successfully.")
