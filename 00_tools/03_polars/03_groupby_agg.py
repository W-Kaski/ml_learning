"""
03_polars / 03_groupby_agg.py
================================
Topic: Grouping and aggregation patterns in Polars.

Covers:
  1. Basic group_by + agg
  2. Multiple aggregations per column
  3. Named custom aggregations
  4. Conditional aggregations (filter inside agg)
  5. over() – window / groupby-transform (like pandas transform)
  6. Dynamic group_by (time-based)
  7. group_by_rolling (rolling window per group)
  8. Pivot after group_by
"""

import polars as pl
import numpy as np

np.random.seed(13)

n = 600
df = pl.DataFrame({
    "emp_id":   list(range(1, n + 1)),
    "dept":     np.random.choice(["Eng", "HR", "Sales", "Finance"], n).tolist(),
    "level":    np.random.choice(["Junior", "Mid", "Senior"], n, p=[0.35, 0.40, 0.25]).tolist(),
    "country":  np.random.choice(["US", "UK", "DE"], n, p=[0.5, 0.3, 0.2]).tolist(),
    "salary":   np.random.randint(40000, 160000, n).tolist(),
    "bonus":    np.random.randint(0, 30000, n).tolist(),
    "rating":   np.round(np.random.uniform(1, 5, n), 1).tolist(),
    "order_date": pl.date_range(
        pl.date(2024, 1, 1), pl.date(2024, 12, 31),
        interval="1d", eager=True
    ).sample(n, with_replacement=True, seed=13).to_list(),
})

# ──────────────────────────────────────────────────────────────
# 1. Basic group_by + agg
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. BASIC GROUP_BY + AGG")
print("=" * 60)

result = (
    df.group_by("dept")
    .agg(pl.col("salary").mean().round(0).alias("avg_salary"))
    .sort("avg_salary", descending=True)
)
print(result)

# ──────────────────────────────────────────────────────────────
# 2. Multiple aggregations per column
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. MULTIPLE AGGREGATIONS")
print("=" * 60)

result = (
    df.group_by("dept")
    .agg([
        pl.col("salary").mean().round(0).alias("avg_salary"),
        pl.col("salary").median().alias("med_salary"),
        pl.col("salary").std().round(0).alias("std_salary"),
        pl.col("salary").min().alias("min_salary"),
        pl.col("salary").max().alias("max_salary"),
        pl.col("bonus").sum().alias("total_bonus"),
        pl.col("rating").mean().round(3).alias("avg_rating"),
        pl.len().alias("headcount"),
    ])
    .sort("avg_salary", descending=True)
)
print(result)

# ──────────────────────────────────────────────────────────────
# 3. Named custom aggregations (groupby multiple keys)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MULTI-KEY GROUP BY")
print("=" * 60)

grid = (
    df.group_by(["dept", "level"])
    .agg([
        pl.col("salary").mean().round(0).alias("avg_salary"),
        pl.len().alias("n"),
    ])
    .sort(["dept", "level"])
)
print(grid)

# ──────────────────────────────────────────────────────────────
# 4. Conditional aggregations (filter inside agg)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CONDITIONAL AGGREGATIONS")
print("=" * 60)

# Count and avg salary of high performers (rating >= 4) per dept
result = (
    df.group_by("dept")
    .agg([
        pl.len().alias("total"),
        pl.col("salary").filter(pl.col("rating") >= 4.0).mean().round(0).alias("avg_salary_high_perf"),
        pl.col("rating").filter(pl.col("rating") >= 4.0).len().alias("n_high_perf"),
        (pl.col("salary").filter(pl.col("rating") >= 4.0).len() / pl.len()).round(3).alias("high_perf_rate"),
    ])
    .sort("high_perf_rate", descending=True)
)
print(result)

# ──────────────────────────────────────────────────────────────
# 5. over() – window / group-transform (broadcast back)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. over() – WINDOW EXPRESSIONS (like pandas transform)")
print("=" * 60)

df_win = df.with_columns([
    pl.col("salary").mean().over("dept").round(0).alias("dept_avg_salary"),
    pl.col("salary").max().over("dept").alias("dept_max_salary"),
    pl.col("salary").rank(descending=True).over("dept").alias("salary_rank_in_dept"),
])

df_win = df_win.with_columns(
    (pl.col("salary") - pl.col("dept_avg_salary")).round(0).alias("salary_vs_dept_avg")
)

print("Sample (first 10 rows, key columns):")
print(df_win.select([
    "emp_id", "dept", "salary",
    "dept_avg_salary", "salary_vs_dept_avg", "salary_rank_in_dept"
]).head(10))

# over with multiple partition keys
df_win2 = df.with_columns(
    pl.col("salary").mean().over(["dept", "level"]).round(0).alias("dept_level_avg")
)
print("\nSample with dept+level partition:")
print(df_win2.select(["emp_id", "dept", "level", "salary", "dept_level_avg"]).head(6))

# ──────────────────────────────────────────────────────────────
# 6. group_by_dynamic – time-based grouping
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. GROUP_BY_DYNAMIC – monthly salary & bonus totals")
print("=" * 60)

df_ts = (
    df.with_columns(pl.col("order_date").cast(pl.Date))
      .sort("order_date")
)

monthly = (
    df_ts.group_by_dynamic("order_date", every="1mo")
    .agg([
        pl.col("salary").mean().round(0).alias("avg_salary"),
        pl.col("bonus").sum().alias("total_bonus"),
        pl.len().alias("n_employees"),
    ])
)
print("Monthly aggregation (first 6):")
print(monthly.head(6))

# ──────────────────────────────────────────────────────────────
# 7. Rolling window per group using group_by + sort + over
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. ROLLING MEAN (sort + shift + over pattern)")
print("=" * 60)

# Create a small ordered time-series per dept
ts_small = pl.DataFrame({
    "dept":    ["Eng"] * 6 + ["Sales"] * 6,
    "month":   list(range(1, 7)) + list(range(1, 7)),
    "revenue": [100, 120, 115, 130, 125, 140,
                 80,  90,  85,  95,  88, 100],
})

ts_rolled = ts_small.sort(["dept", "month"]).with_columns(
    pl.col("revenue")
      .rolling_mean(window_size=3, min_samples=1)
      .over("dept")
      .round(2)
      .alias("rolling_mean_3m")
)
print(ts_rolled)

# ──────────────────────────────────────────────────────────────
# 8. Pivot after group_by
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. PIVOT – dept × level average salary grid")
print("=" * 60)

agg = (
    df.group_by(["dept", "level"])
    .agg(pl.col("salary").mean().round(0).alias("avg_salary"))
)

pivoted = agg.pivot(
    values="avg_salary",
    index="dept",
    on="level",
).sort("dept")
print(pivoted)

# Unpivot back to long
unpivoted = pivoted.unpivot(
    index="dept",
    variable_name="level",
    value_name="avg_salary",
).sort(["dept", "level"])
print("\nUnpivoted (long form):")
print(unpivoted.head(8))

print("\n[Done] 03_groupby_agg.py completed successfully.")
