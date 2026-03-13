"""
03_polars / 02_lazy_api.py
============================
Topic: Polars Lazy API – query planning, optimization, and collect.

Covers:
  1. Eager vs Lazy: when to use each
  2. Creating a LazyFrame (scan_csv, .lazy())
  3. Building a lazy query plan
  4. .explain() – view the optimized query plan
  5. .collect() – execute and materialize
  6. Predicate pushdown (filter early)
  7. Projection pushdown (select only needed columns)
  8. Common lazy pipeline patterns
  9. Streaming mode for larger-than-memory data
"""

import os
import polars as pl
import numpy as np

np.random.seed(77)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Generate a CSV file for scan_csv demos
# ──────────────────────────────────────────────────────────────
N = 200_000
csv_path = f"{DATA_DIR}/employees_large.csv"

if not os.path.exists(csv_path):
    print("Generating large CSV…", end=" ", flush=True)
    df_gen = pl.DataFrame({
        "emp_id":   list(range(1, N + 1)),
        "dept":     np.random.choice(["Eng","HR","Sales","Finance","Legal"], N).tolist(),
        "country":  np.random.choice(["US","UK","DE","FR","CA"], N,
                                      p=[0.4, 0.2, 0.15, 0.15, 0.1]).tolist(),
        "salary":   np.random.randint(30000, 180000, N).tolist(),
        "age":      np.random.randint(20, 65, N).tolist(),
        "years_exp":np.random.randint(0, 30, N).tolist(),
        "rating":   np.round(np.random.uniform(1, 5, N), 1).tolist(),
    })
    df_gen.write_csv(csv_path)
    print(f"done ({N:,} rows → {os.path.getsize(csv_path)//1024} KB)")

# ──────────────────────────────────────────────────────────────
# 1. Eager vs Lazy
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. EAGER vs LAZY")
print("=" * 60)

# Eager: immediate execution
small = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
eager_result = small.filter(pl.col("x") > 2)   # runs NOW
print("Eager filter result:", eager_result["x"].to_list())

# Lazy: deferred execution
lazy_query = small.lazy().filter(pl.col("x") > 2)   # plan only
print(f"Lazy type: {type(lazy_query)}")
lazy_result = lazy_query.collect()    # execute NOW
print("Lazy collect result:", lazy_result["x"].to_list())

print("\nKey difference:")
print("  .lazy() → builds a LogicalPlan (no data moved)")
print("  .collect() → optimizes and executes the plan")

# ──────────────────────────────────────────────────────────────
# 2. scan_csv – lazy file reading
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. scan_csv – LAZY FILE READING")
print("=" * 60)

# scan_csv never reads the full file until .collect()
lf = pl.scan_csv(csv_path)
print(f"LazyFrame type  : {type(lf)}")
print(f"Schema (inferred without reading data):")
for col, dtype in lf.collect_schema().items():
    print(f"  {col:<12} → {dtype}")

# ──────────────────────────────────────────────────────────────
# 3. Building a lazy query plan
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. BUILDING A LAZY QUERY PLAN")
print("=" * 60)

# Build the query step by step (nothing executes yet)
query = (
    pl.scan_csv(csv_path)
    .filter(pl.col("country") == "US")
    .filter(pl.col("salary") > 80000)
    .select(["emp_id", "dept", "salary", "age", "years_exp"])
    .with_columns([
        (pl.col("salary") / 1000).round(2).alias("salary_k"),
        (pl.col("salary") / pl.col("years_exp").clip(lower_bound=1)).round(2).alias("efficiency"),
    ])
    .group_by("dept")
    .agg([
        pl.col("salary_k").mean().round(2).alias("avg_salary_k"),
        pl.col("efficiency").mean().round(2).alias("avg_efficiency"),
        pl.len().alias("headcount"),
    ])
    .sort("avg_salary_k", descending=True)
)
print("Query plan built (no data read yet).")
print(f"Type: {type(query)}")

# ──────────────────────────────────────────────────────────────
# 4. .explain() – view optimized plan
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. EXPLAIN – OPTIMIZED QUERY PLAN")
print("=" * 60)
print(query.explain())

# ──────────────────────────────────────────────────────────────
# 5. .collect() – execute and materialize
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. COLLECT – EXECUTE THE PLAN")
print("=" * 60)

import time
t0 = time.perf_counter()
result = query.collect()
t1 = time.perf_counter()
print(f"Collected in {(t1-t0)*1000:.1f} ms  ({N:,} rows scanned)")
print(result)

# ──────────────────────────────────────────────────────────────
# 6. Predicate pushdown
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. PREDICATE PUSHDOWN (filter moves to scan)")
print("=" * 60)

# Polars automatically pushes the filter as close to the source as possible
# You can observe this in the explain() output
lf_push = (
    pl.scan_csv(csv_path)
    .select(["dept", "salary", "country"])     # projection first in code…
    .filter(pl.col("country") == "DE")         # …filter second in code
)
print("Explain (filter should be pushed before projection):")
print(lf_push.explain())    # optimizer re-orders for efficiency

pushed_result = lf_push.collect()
print(f"\nDE employees: {pushed_result.height:,} rows")
print(pushed_result.head(5))

# ──────────────────────────────────────────────────────────────
# 7. Projection pushdown (only read necessary columns)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. PROJECTION PUSHDOWN (read only needed columns)")
print("=" * 60)

# Only dept + salary will be read from CSV
lf_proj = (
    pl.scan_csv(csv_path)
    .select(["dept", "salary"])
    .filter(pl.col("salary") > 100000)
)
print("Explain (only 2 columns projected at scan level):")
print(lf_proj.explain())
proj_result = lf_proj.collect()
print(f"\nHigh earners: {proj_result.height:,} rows")

# ──────────────────────────────────────────────────────────────
# 8. Common lazy pipeline patterns
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. COMMON LAZY PIPELINE PATTERNS")
print("=" * 60)

# Pattern A: filter → aggregate → rename
pattern_a = (
    pl.scan_csv(csv_path)
    .filter(pl.col("age").is_between(25, 45))
    .group_by("dept")
    .agg(
        pl.col("salary").mean().round(0).alias("avg_salary"),
        pl.len().alias("n"),
    )
    .rename({"dept": "department"})
    .sort("avg_salary", descending=True)
    .collect()
)
print("A: Filter age 25-45, group by dept:")
print(pattern_a)

# Pattern B: join on lazy frames
lf1 = pl.scan_csv(csv_path).select(["emp_id", "dept", "salary"])
dept_targets = pl.LazyFrame({
    "dept":   ["Eng", "HR", "Sales", "Finance", "Legal"],
    "target": [100000, 70000, 75000, 90000, 85000],
})
pattern_b = (
    lf1.join(dept_targets, on="dept", how="left")
       .with_columns(
           (pl.col("salary") / pl.col("target") - 1).round(4).alias("vs_target_pct")
       )
       .group_by("dept")
       .agg(
           pl.col("vs_target_pct").mean().round(4).alias("avg_vs_target"),
           pl.col("vs_target_pct").filter(pl.col("vs_target_pct") > 0).len().alias("above_target_n"),
           pl.len().alias("total"),
       )
       .sort("avg_vs_target", descending=True)
       .collect()
)
print("\nB: Salary vs dept target:")
print(pattern_b)

# ──────────────────────────────────────────────────────────────
# 9. Streaming (larger-than-memory hint)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. STREAMING MODE  (.collect(streaming=True))")
print("=" * 60)

stream_query = (
    pl.scan_csv(csv_path)
    .filter(pl.col("rating") >= 4.0)
    .group_by("country")
    .agg(pl.col("salary").median().round(0).alias("median_salary"), pl.len().alias("n"))
    .sort("median_salary", descending=True)
)

# streaming=True processes in batches — good for huge files
stream_result = stream_query.collect(engine="streaming")
print("Streaming result (high-rated employees by country):")
print(stream_result)
print("\nNote: streaming=True processes data in row batches to limit memory peak.")

print("\n[Done] 02_lazy_api.py completed successfully.")
