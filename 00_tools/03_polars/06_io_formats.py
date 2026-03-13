"""
03_polars / 06_io_formats.py
==============================
Topic: Reading and writing data in various file formats.

Covers:
  1. CSV – read_csv / scan_csv / write_csv  (schema overrides, null inference)
  2. Parquet – read/scan/write, compression, column pruning
  3. JSON / NDJSON – read_json, write_json, scan_ndjson
  4. Schema inference control – dtype overrides, try_parse_dates, infer_schema_length
  5. Detecting and handling bad rows
"""

import polars as pl
import os
import json
import datetime as dt
import tempfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Build a sample dataset
# ──────────────────────────────────────────────────────────────
sample = pl.DataFrame({
    "id":        list(range(1, 101)),
    "name":      [f"Person_{i}" for i in range(1, 101)],
    "age":       ([25, 32, 41, 28, 55] * 20),
    "salary":    ([50_000.0, 75_000.0, 120_000.0, 45_000.0, 95_000.0] * 20),
    "dept":      (["Eng", "HR", "Sales", "Finance", "Legal"] * 20),
    "join_date": pl.date_range(
        dt.date(2020, 1, 1), dt.date(2020, 4, 9), interval="1d", eager=True
    ).to_list(),
    "active":    ([True, True, False, True, False] * 20),
})

# ──────────────────────────────────────────────────────────────
# 1. CSV
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. CSV")
print("=" * 60)

csv_path = os.path.join(DATA_DIR, "sample_io.csv")

# Write
sample.write_csv(csv_path)
print(f"Written: {csv_path}  ({os.path.getsize(csv_path):,} bytes)")

# Eager read – default schema inference
df_csv = pl.read_csv(csv_path)
print("\nDefault schema after read_csv:")
print(df_csv.schema)

# Explicit dtypes + date parsing
df_csv2 = pl.read_csv(
    csv_path,
    schema_overrides={
        "salary": pl.Float64,
        "active": pl.Boolean,
    },
    try_parse_dates=True,
)
print("\nWith try_parse_dates + overrides:")
print(df_csv2.schema)
print(df_csv2.head(3))

# Lazy scan – only read needed columns
lf = (
    pl.scan_csv(csv_path, try_parse_dates=True)
    .filter(pl.col("active") == True)  # noqa: E712
    .select(["id", "name", "dept", "salary"])
    .sort("salary", descending=True)
)
print("\nLazy scan (active employees, top 5 by salary):")
print(lf.head(5).collect())

# ──────────────────────────────────────────────────────────────
# 2. Parquet
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PARQUET")
print("=" * 60)

parquet_path    = os.path.join(DATA_DIR, "sample_io.parquet")
parquet_snappy  = os.path.join(DATA_DIR, "sample_io_snappy.parquet")
parquet_zstd    = os.path.join(DATA_DIR, "sample_io_zstd.parquet")

# Write with different compressions
sample.write_parquet(parquet_path)                                  # default (zstd)
sample.write_parquet(parquet_snappy, compression="snappy")
sample.write_parquet(parquet_zstd,   compression="zstd", compression_level=6)

for label, path in [("default", parquet_path),
                    ("snappy",  parquet_snappy),
                    ("zstd-6",  parquet_zstd)]:
    print(f"  {label:<12}: {os.path.getsize(path):>8,} bytes")

# Eager read – preserves dtypes perfectly (no string/date issues)
df_pq = pl.read_parquet(parquet_path)
print("\nParquet schema (dtypes preserved):")
print(df_pq.schema)

# Lazy scan + column pruning – Parquet only reads requested columns
lf_pq = (
    pl.scan_parquet(parquet_path)
    .filter(pl.col("dept") == "Eng")
    .select(["id", "name", "salary", "join_date"])
)
print("\nParquet lazy scan – Eng dept (column pruning):")
print(lf_pq.collect())

# ──────────────────────────────────────────────────────────────
# 3. JSON and NDJSON
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. JSON / NDJSON")
print("=" * 60)

json_path   = os.path.join(DATA_DIR, "sample_io.json")
ndjson_path = os.path.join(DATA_DIR, "sample_io.ndjson")

small = sample.head(5)

# write_json produces an array-of-objects
small.write_json(json_path)
print(f"Written JSON  : {json_path}  ({os.path.getsize(json_path):,} bytes)")

# read back
df_json = pl.read_json(json_path)
print("Read JSON schema:", df_json.schema)
print(df_json.head(3))

# NDJSON (one JSON object per line) – better for streaming large files
small.write_ndjson(ndjson_path)
print(f"\nWritten NDJSON: {ndjson_path}  ({os.path.getsize(ndjson_path):,} bytes)")

df_ndjson = pl.read_ndjson(ndjson_path)
print("Read NDJSON:", df_ndjson.shape)

# Lazy scan for NDJSON
lf_nj = pl.scan_ndjson(ndjson_path).filter(pl.col("salary") > 60_000)
print("NDJSON lazy (salary > 60k):")
print(lf_nj.collect())

# ──────────────────────────────────────────────────────────────
# 4. Schema overrides and null inference
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SCHEMA OVERRIDES AND NULL INFERENCE")
print("=" * 60)

# Build a tricky CSV manually
tricky_csv = os.path.join(DATA_DIR, "tricky.csv")
with open(tricky_csv, "w") as f:
    f.write("id,value,flag,score,created\n")
    f.write("1,100,1,8.5,2024-01-01\n")
    f.write("2,,0,N/A,2024-01-02\n")        # empty value, N/A score
    f.write("3,200,1,7.2,\n")               # missing date
    f.write("4,N/A,0,9.0,2024-01-04\n")     # N/A value

# Without overrides – everything inferred as strings if mixed
df_raw = pl.read_csv(tricky_csv)
print("Raw inference:")
print(df_raw.schema)
print(df_raw)

# With overrides + custom null values
df_clean = pl.read_csv(
    tricky_csv,
    schema_overrides={"value": pl.Int64, "score": pl.Float64},
    null_values=["N/A", ""],
    try_parse_dates=True,
)
df_clean = df_clean.with_columns(pl.col("flag").cast(pl.Boolean))
print("\nWith overrides + null_values=['N/A', '']:")
print(df_clean.schema)
print(df_clean)

# infer_schema_length: how many rows Polars samples to infer schema
# (useful when schema changes after the first N rows)
df_infer = pl.read_csv(csv_path, infer_schema_length=200)
print(f"\ninfer_schema_length=200 → schema: {df_infer.schema}")

# ──────────────────────────────────────────────────────────────
# 5. Handling bad rows with ignore_errors
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. BAD ROWS (ignore_errors)")
print("=" * 60)

bad_csv = os.path.join(DATA_DIR, "bad_rows.csv")
with open(bad_csv, "w") as f:
    f.write("id,amount\n")
    f.write("1,100\n")
    f.write("2,BROKEN\n")     # non-numeric amount
    f.write("3,300\n")
    f.write("4,\n")           # empty amount → null
    f.write("5,500\n")

# strict=True (default) → raises on bad cast
# Use schema_overrides + null_values to handle gracefully
df_bad = pl.read_csv(
    bad_csv,
    schema_overrides={"amount": pl.Int64},
    null_values=["BROKEN", ""],
    ignore_errors=True,
)
print("Bad rows CSV:")
print(df_bad)
print(f"Null amounts: {df_bad['amount'].null_count()}")

# ──────────────────────────────────────────────────────────────
# Cleanup temp data files
# ──────────────────────────────────────────────────────────────
for p in [parquet_snappy, parquet_zstd, tricky_csv, bad_csv]:
    os.remove(p)

print("\n[Done] 06_io_formats.py completed successfully.")
