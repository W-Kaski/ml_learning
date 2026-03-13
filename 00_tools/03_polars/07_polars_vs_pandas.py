"""
03_polars / 07_polars_vs_pandas.py
=====================================
Topic: Head-to-head comparison of Polars vs pandas.

Covers:
  1. API translation table (printed as reference)
  2. Performance benchmark – filter, groupby, join on 500k rows
  3. Key design differences: eager vs lazy, no index, expression API
  4. When to prefer Polars vs pandas
"""

import time
import polars as pl
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 500_000

# ──────────────────────────────────────────────────────────────
# 1. API translation reference
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. API TRANSLATION REFERENCE")
print("=" * 60)

translations = [
    ("pandas",                                   "polars"),
    ("─" * 40,                                   "─" * 40),
    ("pd.DataFrame({'a': s})",                   "pl.DataFrame({'a': s})"),
    ("df['col'] / df.col",                       "df['col'] (no .col attribute)"),
    ("df[df['x'] > 5]",                          "df.filter(pl.col('x') > 5)"),
    ("df[['a','b']]",                             "df.select(['a','b'])"),
    ("df.assign(c=df.a+1)",                      "df.with_columns((pl.col('a')+1).alias('c'))"),
    ("df.groupby('k').agg({'v':'sum'})",          "df.group_by('k').agg(pl.col('v').sum())"),
    ("df.merge(df2, on='k', how='left')",         "df.join(df2, on='k', how='left')"),
    ("df.rename({'a':'b'}, axis=1)",              "df.rename({'a':'b'})"),
    ("df.drop(columns=['a'])",                    "df.drop(['a'])"),
    ("df.sort_values('a', ascending=False)",      "df.sort('a', descending=True)"),
    ("df.isna() / df.notna()",                    "pl.col('a').is_null() / .is_not_null()"),
    ("df.fillna(0)",                              "df.fill_null(0)"),
    ("df.dropna()",                               "df.drop_nulls()"),
    ("df.apply(func, axis=1)",                    "df.map_rows(func)  (avoid if possible)"),
    ("df.melt(id_vars=['a'])",                    "df.unpivot(index=['a'])"),
    ("df.pivot_table(...)",                       "df.pivot(...)"),
    ("pd.concat([a,b])",                          "pl.concat([a,b])"),
    ("pd.concat([a,b], axis=1)",                  "pl.concat([a,b], how='horizontal')"),
    ("df.dtypes / df.schema",                     "df.schema (OrderedDict)"),
    ("df.shape",                                  "df.shape  (same)"),
    ("df.head(n)",                                "df.head(n)  (same)"),
    ("df.describe()",                             "df.describe()  (same)"),
    ("df.value_counts('col')",                    "df['col'].value_counts()"),
    ("df.index",                                  "No index in Polars (row number via pl.int_range)"),
    ("df.reset_index()",                          "Not needed – Polars has no index"),
    ("df.iterrows()",                             "df.iter_rows() / df.to_dicts() (avoid generally)"),
]

col_w = 42
for left, right in translations:
    print(f"  {left:<{col_w}}  {right}")

# ──────────────────────────────────────────────────────────────
# Build identical datasets in both libraries
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PERFORMANCE BENCHMARK  (N = {:,})".format(N))
print("=" * 60)

depts  = ["Eng", "HR", "Sales", "Finance", "Legal"]
cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix"]

data = {
    "id":     np.arange(N),
    "dept":   np.random.choice(depts,  N),
    "city":   np.random.choice(cities, N),
    "salary": np.random.randint(40_000, 150_000, N).astype(float),
    "age":    np.random.randint(22, 65, N),
    "score":  np.random.random(N).round(4),
}

df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)

lookup_pd = pd.DataFrame({"dept": depts, "budget": [5e6, 2e6, 3e6, 2.5e6, 1.5e6]})
lookup_pl = pl.DataFrame({"dept": depts, "budget": [5e6, 2e6, 3e6, 2.5e6, 1.5e6]})

def bench(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    n = len(result) if hasattr(result, "__len__") else "?"
    print(f"  {label:<30}: {elapsed*1000:7.1f} ms  (rows={n})")
    return elapsed

# — FILTER ——————————————————————————————————————————————————
print("\nFILTER  (salary > 100k AND dept == 'Eng'):")
t_pd = bench("pandas  filter", lambda: df_pd[(df_pd["salary"] > 100_000) & (df_pd["dept"] == "Eng")])
t_pl = bench("polars  filter", lambda: df_pl.filter((pl.col("salary") > 100_000) & (pl.col("dept") == "Eng")))
print(f"  → Polars speedup: {t_pd/t_pl:.1f}x")

# — GROUP BY ————————————————————————————————————————————————
print("\nGROUP BY dept + agg (mean salary, max score, count):")
t_pd = bench("pandas  groupby", lambda: df_pd.groupby("dept", as_index=False).agg(
    avg_salary=("salary", "mean"),
    max_score=("score", "max"),
    n=("id", "count"),
))
t_pl = bench("polars  group_by", lambda: df_pl.group_by("dept").agg([
    pl.col("salary").mean().round(0).alias("avg_salary"),
    pl.col("score").max().alias("max_score"),
    pl.len().alias("n"),
]))
print(f"  → Polars speedup: {t_pd/t_pl:.1f}x")

# — JOIN ————————————————————————————————————————————————————
print("\nJOIN (left join on dept):")
t_pd = bench("pandas  merge",   lambda: df_pd.merge(lookup_pd, on="dept", how="left"))
t_pl = bench("polars  join",    lambda: df_pl.join(lookup_pl,  on="dept", how="left"))
print(f"  → Polars speedup: {t_pd/t_pl:.1f}x")

# — CHAIN (filter → groupby → sort) ————————————————————————
print("\nCHAIN: filter + groupby + sort:")
t_pd = bench("pandas  chain", lambda: (
    df_pd[df_pd["age"] < 40]
    .groupby(["dept", "city"], as_index=False)
    .agg(avg_sal=("salary", "mean"), n=("id", "count"))
    .sort_values("avg_sal", ascending=False)
))
t_pl = bench("polars  chain", lambda: (
    df_pl
    .filter(pl.col("age") < 40)
    .group_by(["dept", "city"])
    .agg([pl.col("salary").mean().alias("avg_sal"), pl.len().alias("n")])
    .sort("avg_sal", descending=True)
))
print(f"  → Polars speedup: {t_pd/t_pl:.1f}x")

# ──────────────────────────────────────────────────────────────
# 3. Design difference highlights
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. KEY DESIGN DIFFERENCES")
print("=" * 60)

print("""
  EAGER vs LAZY
    pandas:  always eager (executes immediately)
    polars:  optional lazy API – .lazy() → query plan → .collect()
             Lazy enables predicate/projection pushdown and query fusion.

  INDEX
    pandas:  row index (Int64, string, MultiIndex) → .loc/.iloc
    polars:  NO index – rows are always 0-based positional
             Use pl.int_range(pl.len()) to generate a row number column.

  EXPRESSION API
    pandas:  operate on Series objects returned from []
    polars:  expressions (pl.col('x') + 1) are composable and lazy;
             executed inside select / with_columns / filter / agg.

  MUTABILITY
    pandas:  DataFrames are mutable (df['x'] = ...)
    polars:  DataFrames are immutable – every operation returns a new DF.

  NULL vs NaN
    pandas:  NaN for both missing floats AND missing non-numeric
    polars:  null  = missing value (any dtype)
             NaN   = IEEE float NaN (distinct from null)
             pl.col('x').is_null() vs .is_nan()

  MEMORY
    polars:  Apache Arrow columnar format (zero-copy slices, SIMD ops)
    pandas:  NumPy-backed (non-contiguous in memory for object columns)
""")

# ──────────────────────────────────────────────────────────────
# 4. When to prefer each
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("4. WHEN TO PREFER POLARS vs PANDAS")
print("=" * 60)

print("""
  USE POLARS when:
    - Data > a few hundred thousand rows (speed matters a lot)
    - Reading large CSVs/Parquet (scan_csv, scan_parquet + lazy)
    - Building data pipelines (composable expression chains)
    - Memory is constrained (Arrow is more efficient)
    - You want correctness: strict null/NaN, no silent dtype coercion

  USE PANDAS when:
    - Ecosystem compatibility required (sklearn, statsmodels, plotly…)
    - Time series: DatetimeIndex-based resampling, rolling (pandas excels)
    - Interactive exploration: .plot(), Jupyter repr, rich iPython display
    - Small datasets where boilerplate cost matters more than speed
    - Code that must interact with pd.DataFrame as input/output API
""")

print("[Done] 07_polars_vs_pandas.py completed successfully.")
