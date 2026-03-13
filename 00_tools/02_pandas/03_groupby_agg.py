"""
02_pandas / 03_groupby_agg.py
================================
Topic: Grouping data and computing aggregations in pandas.

Covers:
  1. Basic groupby + single aggregation
  2. Multiple aggregations at once (agg with dict)
  3. Named aggregations (pandas ≥ 0.25 style)
  4. transform – broadcast group stats back to original index
  5. filter – keep/drop entire groups by condition
  6. apply – arbitrary group-level function
  7. Hierarchical groupby (multi-level groups)
  8. Resample (time-based groupby)
"""

import numpy as np
import pandas as pd

np.random.seed(7)

# ──────────────────────────────────────────────────────────────
# 1. Build dataset
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATASET: EMPLOYEE RECORDS")
print("=" * 60)

n = 400
departments = np.random.choice(["Engineering", "Sales", "Marketing", "HR"], n,
                                p=[0.40, 0.30, 0.20, 0.10])
levels      = np.random.choice(["Junior", "Mid", "Senior"], n, p=[0.35, 0.40, 0.25])
salaries    = (
    np.where(levels == "Junior",  np.random.normal(55000, 8000, n),
    np.where(levels == "Mid",     np.random.normal(80000, 10000, n),
                                   np.random.normal(110000, 15000, n)))
).clip(30000, 200000).round(2)

df = pd.DataFrame({
    "employee_id": range(1, n + 1),
    "department":  departments,
    "level":       levels,
    "salary":      salaries,
    "years_exp":   np.random.randint(0, 20, n),
    "rating":      np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.25, 0.40, 0.20]),
})

print(df.head(8).to_string(index=False))
print(f"\nShape: {df.shape}")

# ──────────────────────────────────────────────────────────────
# 2. Basic groupby + single aggregation
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. BASIC GROUPBY – mean salary by department")
print("=" * 60)

dept_mean = df.groupby("department")["salary"].mean().round(2).sort_values(ascending=False)
print(dept_mean.to_string())

# Group size (shorthand: value_counts)
print("\nGroup sizes:")
print(df.groupby("department").size().to_string())

# ──────────────────────────────────────────────────────────────
# 3. Multiple aggregations
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MULTIPLE AGGREGATIONS (.agg with dict)")
print("=" * 60)

dept_stats = df.groupby("department").agg(
    salary_mean   = ("salary",   "mean"),
    salary_median = ("salary",   "median"),
    salary_std    = ("salary",   "std"),
    headcount     = ("salary",   "count"),
    avg_rating    = ("rating",   "mean"),
    avg_exp       = ("years_exp","mean"),
).round(2)
print(dept_stats.to_string())

# ──────────────────────────────────────────────────────────────
# 4. Named aggregations (NamedAgg) — identical to above, cleaner syntax
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. NAMED AGG – top earner & min earner per department")
print("=" * 60)

extremes = df.groupby("department").agg(
    top_salary = pd.NamedAgg(column="salary", aggfunc="max"),
    min_salary = pd.NamedAgg(column="salary", aggfunc="min"),
    top_emp    = pd.NamedAgg(column="employee_id",
                              aggfunc=lambda x: x.iloc[df.loc[x.index, "salary"].argmax()]),
).round(2)
print(extremes.to_string())

# ──────────────────────────────────────────────────────────────
# 5. transform – add group mean as new column (same index)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TRANSFORM – add dept mean salary & salary_vs_mean")
print("=" * 60)

df["dept_mean_salary"] = df.groupby("department")["salary"].transform("mean").round(2)
df["salary_vs_mean"]   = (df["salary"] - df["dept_mean_salary"]).round(2)

sample = df.groupby("department").head(2)[
    ["employee_id", "department", "salary", "dept_mean_salary", "salary_vs_mean"]
]
print(sample.to_string(index=False))

# Also useful: z-score within group
df["salary_z"] = (
    df.groupby("department")["salary"]
      .transform(lambda x: (x - x.mean()) / x.std())
      .round(3)
)
print(f"\nsalary_z range: [{df['salary_z'].min():.2f}, {df['salary_z'].max():.2f}]")

# ──────────────────────────────────────────────────────────────
# 6. filter – keep groups meeting a condition
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FILTER – keep departments with avg salary > 75 000")
print("=" * 60)

df_high_pay = df.groupby("department").filter(
    lambda g: g["salary"].mean() > 75_000
)
kept_depts = df_high_pay["department"].unique()
print(f"Departments kept : {sorted(kept_depts)}")
print(f"Rows kept        : {len(df_high_pay)} / {len(df)}")

# ──────────────────────────────────────────────────────────────
# 7. apply – arbitrary group-level computation
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. APPLY – percentile salary bands within each department")
print("=" * 60)

def salary_bands(group):
    p25 = group["salary"].quantile(0.25)
    p75 = group["salary"].quantile(0.75)
    group = group.copy()
    group["band"] = np.where(group["salary"] < p25, "Low",
                    np.where(group["salary"] > p75, "High", "Mid"))
    return group

df_banded = df.groupby("department", group_keys=False).apply(salary_bands, include_groups=False)
df_banded["department"] = df["department"]   # restore after exclude
print(df_banded.groupby(["department", "band"]).size().unstack(fill_value=0).to_string())

# ──────────────────────────────────────────────────────────────
# 8. Hierarchical groupby
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. HIERARCHICAL GROUPBY – dept × level salary grid")
print("=" * 60)

grid = (
    df.groupby(["department", "level"])["salary"]
      .agg(["mean", "count"])
      .round(2)
      .rename(columns={"mean": "avg_salary", "count": "n"})
)
print(grid.to_string())

# Unstack level to wide format
wide = grid["avg_salary"].unstack("level").round(0)
print("\nPivoted (avg salary per dept×level):")
print(wide.to_string())

# ──────────────────────────────────────────────────────────────
# 9. Resample – time-based groupby
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. RESAMPLE – weekly average salary of new hires")
print("=" * 60)

# Simulate hire dates
hire_dates = pd.date_range("2023-01-01", periods=n, freq="D")
df["hire_date"] = np.random.choice(hire_dates, n, replace=False)
df_ts = df.set_index("hire_date").sort_index()

weekly_avg = df_ts["salary"].resample("W").mean().dropna().round(2)
print(f"Weekly data points: {len(weekly_avg)}")
print(weekly_avg.head(8).to_string())

monthly_count = df_ts["salary"].resample("ME").count()
print(f"\nMonthly hire counts (first 6):")
print(monthly_count.head(6).to_string())

print("\n[Done] 03_groupby_agg.py completed successfully.")
