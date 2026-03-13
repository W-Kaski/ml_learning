"""
03_polars / 00_series_dataframe.py
=====================================
Topic: Polars fundamentals – Series, DataFrame creation, and basic ops.

Covers:
  1. Creating Series and DataFrames
  2. Data types (dtype system)
  3. Basic inspection: shape, schema, head/tail, describe
  4. Column selection and filtering (eager API)
  5. Adding / renaming / dropping columns
  6. Sorting, unique, value_counts
  7. Null handling (null vs NaN distinction in Polars)
  8. Polars vs pandas key mindset differences
"""

import polars as pl
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. Creating Series and DataFrames
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. CREATING SERIES AND DATAFRAMES")
print("=" * 60)

# Series
s_int   = pl.Series("ages",    [22, 35, 41, 29, 55])
s_str   = pl.Series("names",   ["Alice", "Bob", "Carol", "Dave", "Eve"])
s_float = pl.Series("scores",  [8.5, 7.2, 9.1, 6.8, 8.9])
s_bool  = pl.Series("active",  [True, False, True, True, False])

print("Series examples:")
print(f"  int   : {s_int.to_list()}")
print(f"  str   : {s_str.to_list()}")
print(f"  float : {s_float.to_list()}")
print(f"  bool  : {s_bool.to_list()}")

# DataFrame
df = pl.DataFrame({
    "name":       ["Alice", "Bob", "Carol", "Dave", "Eve",
                   "Frank", "Grace", "Hank"],
    "age":        [22, 35, 41, 29, 55, 33, 47, 26],
    "department": ["Eng", "HR", "Eng", "Sales", "HR", "Sales", "Eng", "HR"],
    "salary":     [85000, 62000, 92000, 55000, 70000, 60000, 95000, 58000],
    "rating":     [4.2, 3.8, 4.7, 3.5, 4.0, 3.9, 4.5, 3.6],
    "active":     [True, True, True, False, True, False, True, True],
})

print(f"\nDataFrame (shape {df.shape}):")
print(df)

# ──────────────────────────────────────────────────────────────
# 2. Data types
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DTYPE SYSTEM")
print("=" * 60)

print("Schema:")
for col, dtype in df.schema.items():
    print(f"  {col:<12} → {dtype}")

# Casting
df_typed = df.with_columns([
    pl.col("salary").cast(pl.Float64),
    pl.col("age").cast(pl.UInt8),
    pl.col("department").cast(pl.Categorical),
])
print("\nAfter casting – schema:")
for col, dtype in df_typed.schema.items():
    print(f"  {col:<12} → {dtype}")

# ──────────────────────────────────────────────────────────────
# 3. Basic inspection
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. BASIC INSPECTION")
print("=" * 60)

print(f"shape   : {df.shape}")
print(f"height  : {df.height}")
print(f"width   : {df.width}")
print(f"columns : {df.columns}")
print(f"\nhead(3):")
print(df.head(3))
print(f"\ntail(3):")
print(df.tail(3))
print(f"\ndescribe():")
print(df.describe())

# ──────────────────────────────────────────────────────────────
# 4. Column selection and filtering
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SELECTION AND FILTERING")
print("=" * 60)

# Select specific columns
print("Select name + salary:")
print(df.select(["name", "salary"]))

# select with expressions
print("\nSelect age and salary scaled:")
print(df.select([
    pl.col("name"),
    pl.col("age"),
    (pl.col("salary") / 1000).alias("salary_k"),
]))

# Filter rows
print("\nFilter: active=True AND salary > 65000:")
filtered = df.filter(
    (pl.col("active") == True) & (pl.col("salary") > 65000)
)
print(filtered)

# Filter with .is_in()
print("\nFilter: department in [Eng, Sales]:")
print(df.filter(pl.col("department").is_in(["Eng", "Sales"])))

# ──────────────────────────────────────────────────────────────
# 5. Adding / renaming / dropping columns
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. MUTATING COLUMNS")
print("=" * 60)

# with_columns: add new or replace
df2 = df.with_columns([
    (pl.col("salary") * 1.10).alias("salary_raised"),
    (pl.col("age") - 18).alias("years_adult"),
    pl.lit("2024").alias("year"),
])
print("Added salary_raised, years_adult, year:")
print(df2.select(["name", "salary", "salary_raised", "years_adult", "year"]))

# Rename
df3 = df.rename({"rating": "perf_score", "active": "is_active"})
print(f"\nRenamed columns: {df3.columns}")

# Drop
df4 = df.drop(["active", "rating"])
print(f"Dropped columns: {df4.columns}")

# ──────────────────────────────────────────────────────────────
# 6. Sort, unique, value_counts
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SORT / UNIQUE / VALUE_COUNTS")
print("=" * 60)

# Sort
print("Sorted by salary desc:")
print(df.sort("salary", descending=True))

# Multi-column sort
print("\nSorted by department asc, salary desc:")
print(df.sort(["department", "salary"], descending=[False, True]))

# Unique
print(f"\nUnique departments: {df['department'].unique().to_list()}")
print(f"n_unique departments: {df['department'].n_unique()}")

# value_counts
print("\nDepartment value_counts:")
print(df["department"].value_counts().sort("department"))

# ──────────────────────────────────────────────────────────────
# 7. Null handling
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. NULL HANDLING  (Polars uses null, not NaN for missing)")
print("=" * 60)

df_nulls = pl.DataFrame({
    "a": [1, None, 3, None, 5],
    "b": [10.0, 20.0, None, 40.0, None],
    "c": ["x", "y", None, "w", "v"],
})

print("DataFrame with nulls:")
print(df_nulls)

# Detect nulls
print("\nNull counts:")
print(df_nulls.null_count())

# is_null / is_not_null
print("\nRows where 'a' is null:")
print(df_nulls.filter(pl.col("a").is_null()))

# fill_null
df_filled = df_nulls.with_columns([
    pl.col("a").fill_null(0),
    pl.col("b").fill_null(strategy="forward"),
    pl.col("c").fill_null("unknown"),
])
print("\nAfter fill_null:")
print(df_filled)

# drop_nulls
print(f"\nRows after drop_nulls: {df_nulls.drop_nulls().height}")

# NaN is different from null in Polars (only in Float columns)
df_nan = pl.DataFrame({"x": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
print(f"\nNaN count  (is_nan) : {df_nan['x'].is_nan().sum()}")
print(f"Null count (is_null): {df_nan['x'].is_null().sum()}")
df_nan_filled = df_nan.with_columns(pl.col("x").fill_nan(None).fill_null(0.0))
print("After fill_nan(None) → fill_null(0):", df_nan_filled["x"].to_list())

# ──────────────────────────────────────────────────────────────
# 8. Key mindset differences vs pandas
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. POLARS vs PANDAS KEY DIFFERENCES")
print("=" * 60)

diffs = [
    ("Immutability",    "Polars DataFrames are immutable; ops return new DFs"),
    ("No index",        "Polars has no row index (no df.iloc[0])"),
    ("Expressions",     "Operations use expression API: pl.col('x') + 1"),
    ("Lazy evaluation", "Use .lazy() for query planning; .collect() to execute"),
    ("null vs NaN",     "null = missing; NaN = float Not-a-Number (separate)"),
    ("Parallelism",     "Polars auto-parallelizes over columns and groupby ops"),
    ("Memory",          "Arrow-backed: columnar, zero-copy slices where possible"),
    ("Speed",           "Typically 5–30× faster than pandas for large data"),
]
for concept, note in diffs:
    print(f"  {concept:<20} → {note}")

print("\n[Done] 00_series_dataframe.py completed successfully.")
