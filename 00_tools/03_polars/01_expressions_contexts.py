"""
03_polars / 01_expressions_contexts.py
==========================================
Topic: The Polars Expression API and execution contexts.

Covers:
  1. What is an expression (pl.col, pl.lit, pl.when)
  2. Arithmetic and comparison expressions
  3. select context  – compute per-column transformations
  4. with_columns context – add/replace columns
  5. filter context  – row selection
  6. group_by + agg context
  7. Chaining expressions
  8. pl.when / pl.then / pl.otherwise (conditional expressions)
  9. String, list, and struct expressions
"""

import polars as pl

# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
import numpy as np
np.random.seed(5)

n = 500
df = pl.DataFrame({
    "id":         list(range(1, n + 1)),
    "name":       [f"Person_{i}" for i in range(n)],
    "age":        np.random.randint(18, 65, n).tolist(),
    "dept":       np.random.choice(["Eng", "HR", "Sales", "Finance"], n).tolist(),
    "salary":     np.random.randint(40000, 150000, n).tolist(),
    "years_exp":  np.random.randint(0, 25, n).tolist(),
    "rating":     np.round(np.random.uniform(1, 5, n), 1).tolist(),
    "score_a":    np.random.randint(0, 100, n).tolist(),
    "score_b":    np.random.randint(0, 100, n).tolist(),
})

# ──────────────────────────────────────────────────────────────
# 1. What is an expression
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. EXPRESSIONS  (lazy references to column ops)")
print("=" * 60)

# An expression is a recipe — it only executes inside a context
expr_salary_k = pl.col("salary") / 1000
expr_combo    = (pl.col("score_a") + pl.col("score_b")) / 2

# Execute inside select
result = df.select([
    pl.col("name"),
    expr_salary_k.alias("salary_k"),
    expr_combo.alias("avg_score"),
])
print(result.head(5))

# pl.all(), pl.first(), pl.last()
print("\nColumn names matching 'score':",
      df.select(pl.col("^score.*$")).columns)

# ──────────────────────────────────────────────────────────────
# 2. Arithmetic and comparison expressions
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ARITHMETIC AND COMPARISON EXPRESSIONS")
print("=" * 60)

result = df.select([
    pl.col("name"),
    pl.col("salary"),
    (pl.col("salary") * 1.10).round(0).alias("salary_10pct_raise"),
    (pl.col("salary") / pl.col("years_exp").clip(lower_bound=1)).round(2).alias("salary_per_year_exp"),
    (pl.col("score_a") > pl.col("score_b")).alias("a_beats_b"),
    (pl.col("age").is_between(30, 50)).alias("prime_age"),
])
print(result.head(8))

# ──────────────────────────────────────────────────────────────
# 3. select context
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. SELECT CONTEXT")
print("=" * 60)

# select returns a DataFrame with *only* the listed columns/expressions
# Useful for transformation without modifying original

# Global statistics in select (broadcast with pl.lit)
stats = df.select([
    pl.col("salary").mean().alias("mean_salary"),
    pl.col("salary").std().alias("std_salary"),
    pl.col("salary").min().alias("min_salary"),
    pl.col("salary").max().alias("max_salary"),
    pl.col("age").median().alias("median_age"),
])
print("Global stats:")
print(stats)

# Select with multiple column patterns
print("\nAll columns except 'name' and 'id':")
print(df.select(pl.all().exclude(["name", "id"])).head(3))

# ──────────────────────────────────────────────────────────────
# 4. with_columns context
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. WITH_COLUMNS CONTEXT")
print("=" * 60)

# with_columns keeps ALL original cols and adds/replaces
df2 = df.with_columns([
    (pl.col("salary") / 1000).alias("salary_k"),
    (pl.col("score_a") + pl.col("score_b")).alias("total_score"),
    pl.col("name").str.to_uppercase().alias("name_upper"),
    pl.col("dept").cast(pl.Categorical),
])
print("After with_columns:")
print(df2.select(["name", "salary_k", "total_score", "name_upper", "dept"]).head(5))
print(f"Columns: {df2.columns}")

# ──────────────────────────────────────────────────────────────
# 5. filter context
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. FILTER CONTEXT")
print("=" * 60)

# Multiple conditions
result = df.filter(
    (pl.col("dept") == "Eng") &
    (pl.col("salary") > 90000) &
    (pl.col("years_exp") >= 5)
)
print(f"Eng employees with salary>90k and exp>=5: {result.height} rows")
print(result.head(4))

# filter with is_in
print("\nHR or Finance employees:")
print(df.filter(pl.col("dept").is_in(["HR", "Finance"])).head(4))

# filter with str ops
print("\nNames starting with 'Person_1' (first 10 in list):")
print(df.filter(pl.col("name").str.starts_with("Person_1")).head(3))

# ──────────────────────────────────────────────────────────────
# 6. group_by + agg context
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. GROUP_BY + AGG CONTEXT")
print("=" * 60)

dept_stats = (
    df.group_by("dept")
    .agg([
        pl.col("salary").mean().round(2).alias("avg_salary"),
        pl.col("salary").std().round(2).alias("std_salary"),
        pl.col("salary").max().alias("max_salary"),
        pl.col("rating").mean().round(3).alias("avg_rating"),
        pl.len().alias("headcount"),
        pl.col("years_exp").sum().alias("total_exp_years"),
    ])
    .sort("avg_salary", descending=True)
)
print(dept_stats)

# Multiple group-by keys
multi_grp = (
    df.with_columns(
        pl.when(pl.col("salary") >= 90000).then(pl.lit("High"))
          .otherwise(pl.lit("Low")).alias("pay_band")
    )
    .group_by(["dept", "pay_band"])
    .agg(pl.len().alias("count"), pl.col("salary").mean().round(0).alias("avg_salary"))
    .sort(["dept", "pay_band"])
)
print("\nDepart × pay_band:")
print(multi_grp)

# ──────────────────────────────────────────────────────────────
# 7. Chaining expressions
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. CHAINING EXPRESSIONS")
print("=" * 60)

# Full pipeline: filter → enrich → aggregate
result = (
    df
    .filter(pl.col("rating") >= 3.0)
    .with_columns([
        (pl.col("salary") / pl.col("years_exp").clip(lower_bound=1)).alias("salary_per_exp"),
        pl.col("score_a").add(pl.col("score_b")).alias("combined_score"),
    ])
    .group_by("dept")
    .agg([
        pl.col("salary_per_exp").mean().round(2).alias("avg_salary_per_exp"),
        pl.col("combined_score").mean().round(2).alias("avg_combined_score"),
        pl.len().alias("n"),
    ])
    .sort("avg_salary_per_exp", descending=True)
)
print(result)

# ──────────────────────────────────────────────────────────────
# 8. pl.when / pl.then / pl.otherwise
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. CONDITIONAL EXPRESSIONS (pl.when)")
print("=" * 60)

df_cond = df.with_columns([
    pl.when(pl.col("salary") >= 100000).then(pl.lit("Senior"))
      .when(pl.col("salary") >= 70000).then(pl.lit("Mid"))
      .otherwise(pl.lit("Junior")).alias("level"),

    pl.when(pl.col("rating") >= 4.5).then(pl.lit("A"))
      .when(pl.col("rating") >= 3.5).then(pl.lit("B"))
      .when(pl.col("rating") >= 2.5).then(pl.lit("C"))
      .otherwise(pl.lit("D")).alias("grade"),
])
print("Level and grade distribution:")
print(df_cond["level"].value_counts().sort("level"))
print()
print(df_cond["grade"].value_counts().sort("grade"))

# ──────────────────────────────────────────────────────────────
# 9. String, list, and struct expressions
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. STRING / LIST / STRUCT NAMESPACES")
print("=" * 60)

# String namespace: .str.*
df_str = pl.DataFrame({"raw": ["Alice Smith", "bob jones", "CAROL WHITE", "dave BROWN"]})
df_str = df_str.with_columns([
    pl.col("raw").str.to_titlecase().alias("titlecase"),
    pl.col("raw").str.to_uppercase().alias("upper"),
    pl.col("raw").str.len_chars().alias("char_len"),
    pl.col("raw").str.split(" ").list.get(0).alias("first_name"),
    pl.col("raw").str.contains("smith", literal=False).alias("is_smith"),
])
print("String ops:")
print(df_str)

# List namespace: .list.*
df_list = pl.DataFrame({"vals": [[1, 2, 3], [10, 20], [5, 5, 5, 5]]})
df_list = df_list.with_columns([
    pl.col("vals").list.len().alias("list_len"),
    pl.col("vals").list.sum().alias("list_sum"),
    pl.col("vals").list.mean().alias("list_mean"),
    pl.col("vals").list.max().alias("list_max"),
])
print("\nList ops:")
print(df_list)

print("\n[Done] 01_expressions_contexts.py completed successfully.")
