"""
03_polars / 08_polars_mini_project.py
=======================================
Mini-project: End-to-end HR analytics pipeline.

Pipeline stages:
  1. Ingest   – scan_csv lazy, validate schema
  2. Clean    – nulls, outliers, dedup
  3. Enrich   – tenure, salary band, performance tier
  4. Aggregate – dept summary, city × dept pivot
  5. Export   – Parquet + CSV report
  6. Insights  – top earners, risk dashboard

Uses the lazy API throughout; streaming collect for the large file.
"""

import polars as pl
import os
import datetime as dt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "employees_large.csv")

if not os.path.exists(CSV_PATH):
    print("employees_large.csv not found – run 02_lazy_api.py first.")
    raise SystemExit(1)

print("=" * 60)
print("STAGE 1: INGEST & VALIDATE")
print("=" * 60)

lf_raw = pl.scan_csv(CSV_PATH, try_parse_dates=True)

schema = lf_raw.collect_schema()
print("Schema:", schema)
# Actual columns: emp_id, dept, country, salary, age, years_exp, rating

row_count = lf_raw.select(pl.len()).collect().item()
print(f"Rows: {row_count:,}")

# Basic quality summary (collect only what we need)
null_counts = (
    lf_raw
    .select([pl.col(c).is_null().sum().alias(c) for c in schema.names()])
    .collect()
)
print("\nNull counts per column:")
print(null_counts)

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: CLEAN")
print("=" * 60)

lf_clean = (
    lf_raw
    # Remove duplicates on employee_id
    .unique(subset=["emp_id"], keep="first")
    # Drop rows with null salary or dept
    .drop_nulls(subset=["salary", "dept"])
    # Clamp salary outliers (keep between 1st and 99th percentile – approximate)
    .filter(
        (pl.col("salary") >= 20_000) &
        (pl.col("salary") <= 500_000)
    )
)

clean_count = lf_clean.select(pl.len()).collect().item()
print(f"Rows after cleaning: {clean_count:,}  (removed {row_count - clean_count:,})")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3: ENRICH")
print("=" * 60)

lf_enriched = lf_clean.with_columns([
    # Salary band
    pl.when(pl.col("salary") < 60_000).then(pl.lit("Low"))
      .when(pl.col("salary") < 100_000).then(pl.lit("Mid"))
      .when(pl.col("salary") < 150_000).then(pl.lit("High"))
      .otherwise(pl.lit("Executive"))
      .alias("salary_band"),

        # Performance tier based on rating (0.0 – 1.0)
        pl.when(pl.col("rating") >= 0.8).then(pl.lit("Top"))
            .when(pl.col("rating") >= 0.5).then(pl.lit("Solid"))
      .otherwise(pl.lit("Develop"))
      .alias("perf_tier"),
])

# Peek at the enriched data (collect first 5 for inspection)
sample_enriched = lf_enriched.head(5).collect()
print("Sample enriched rows:")
print(sample_enriched.select(["emp_id", "dept", "salary", "years_exp",
                                                             "salary_band", "perf_tier"]))

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 4: AGGREGATE")
print("=" * 60)

# Dept summary
dept_summary = (
    lf_enriched
    .group_by("dept")
    .agg([
        pl.len().alias("headcount"),
        pl.col("salary").mean().round(0).alias("avg_salary"),
        pl.col("salary").median().round(0).alias("median_salary"),
        pl.col("salary").max().alias("max_salary"),
        pl.col("years_exp").mean().round(1).alias("avg_exp"),
        (pl.col("perf_tier") == "Top").sum().alias("top_performers"),
    ])
    .with_columns(
        (pl.col("top_performers") / pl.col("headcount") * 100)
        .round(1)
        .alias("top_pct")
    )
    .sort("avg_salary", descending=True)
    .collect()
)
print("Department summary:")
print(dept_summary)

# Salary band distribution per dept (pivot)
band_dist = (
    lf_enriched
    .group_by(["dept", "salary_band"])
    .agg(pl.len().alias("n"))
    .collect()
    .pivot(index="dept", on="salary_band", values="n")
    .fill_null(0)
)
print("\nSalary band distribution (count per dept):")
print(band_dist)

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 5: EXPORT")
print("=" * 60)

# Save full enriched dataset as Parquet (faster reads later)
enriched_parquet = os.path.join(REPORT_DIR, "employees_enriched.parquet")
(
    lf_enriched
    .collect(engine="streaming")
    .write_parquet(enriched_parquet, compression="zstd")
)
print(f"Enriched Parquet  → {enriched_parquet}  ({os.path.getsize(enriched_parquet):,} bytes)")

# Save dept summary as CSV report
dept_report_path = os.path.join(REPORT_DIR, "dept_summary.csv")
dept_summary.write_csv(dept_report_path)
print(f"Dept summary CSV  → {dept_report_path}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 6: INSIGHTS")
print("=" * 60)

# Now scan the Parquet for downstream queries (much faster than CSV)
lf_pq = pl.scan_parquet(enriched_parquet)

# Top 10 earners across the company
top_earners = (
    lf_pq
    .sort("salary", descending=True)
    .head(10)
    .select(["emp_id", "dept", "salary", "years_exp", "perf_tier"])
    .collect()
)
print("Top 10 earners:")
print(top_earners)

# Retention risk: high salary band + short tenure + Develop tier
at_risk = (
    lf_pq
    .filter(
        (pl.col("salary_band").is_in(["High", "Executive"])) &
        (pl.col("years_exp") < 3) &
        (pl.col("perf_tier") == "Develop")
    )
    .group_by("dept")
    .agg([
        pl.len().alias("at_risk_count"),
        pl.col("salary").mean().round(0).alias("avg_salary"),
    ])
    .sort("at_risk_count", descending=True)
    .collect()
)
print("\nRetention risk (high salary, short tenure, needs development):")
print(at_risk)

# Tenure quartile breakdown
tenure_stats = (
    lf_pq
    .group_by("dept")
    .agg([
        pl.col("years_exp").quantile(0.25).round(1).alias("q1_exp"),
        pl.col("years_exp").quantile(0.50).round(1).alias("median_exp"),
        pl.col("years_exp").quantile(0.75).round(1).alias("q3_exp"),
    ])
    .sort("median_exp")
    .collect()
)
print("\nExperience quartiles by dept:")
print(tenure_stats)

print("\n[Done] 08_polars_mini_project.py completed successfully.")
