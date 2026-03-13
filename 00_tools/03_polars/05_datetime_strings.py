"""
03_polars / 05_datetime_strings.py
=====================================
Topic: Date/time and string manipulation in Polars.

Covers:
  1. Date / Datetime types and construction
  2. .dt namespace – extraction (year, month, day, weekday, hour …)
  3. Duration arithmetic (add/subtract, diff, truncate)
  4. String construction from text columns  (strptime, to_date)
  5. .str namespace – slice, split, contains, replace, extract regex
  6. Combined pipeline example
"""

import polars as pl
import datetime as dt

# ──────────────────────────────────────────────────────────────
# 1. Building Date / Datetime columns
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATE / DATETIME TYPES")
print("=" * 60)

# pl.date_range → Date series
dates = pl.date_range(dt.date(2024, 1, 1), dt.date(2024, 12, 31),
                      interval="1mo", eager=True)
print("Monthly dates:")
print(dates)
print("dtype:", dates.dtype)

# pl.datetime_range → Datetime series
dts = pl.datetime_range(
    dt.datetime(2024, 6, 1, 0, 0, 0),
    dt.datetime(2024, 6, 1, 12, 0, 0),
    interval="2h",
    eager=True,
)
print("\nEvery-2h datetime series:")
print(dts)

# ──────────────────────────────────────────────────────────────
# 2. .dt namespace — extraction
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. .dt NAMESPACE – EXTRACTION")
print("=" * 60)

df_dates = pl.DataFrame({"date": dates})
extracted = df_dates.with_columns([
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.day().alias("day"),
    pl.col("date").dt.weekday().alias("weekday"),   # Mon=1 … Sun=7
    pl.col("date").dt.week().alias("iso_week"),
    pl.col("date").dt.ordinal_day().alias("day_of_year"),
    pl.col("date").dt.quarter().alias("quarter"),
])
print(extracted)

# Floor / truncate to start of month / week
print("\nDate truncation to start of month:")
trunc = df_dates.with_columns(
    pl.col("date").dt.truncate("1mo").alias("month_start"),
    pl.col("date").dt.truncate("1w").alias("week_start"),
)
print(trunc.head(4))

# ──────────────────────────────────────────────────────────────
# 3. Duration arithmetic
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DURATION ARITHMETIC")
print("=" * 60)

events = pl.DataFrame({
    "event_id":  [1, 2, 3, 4],
    "start":     [dt.datetime(2024, 3, 1, 9, 0),
                  dt.datetime(2024, 3, 5, 14, 30),
                  dt.datetime(2024, 3, 10, 8, 0),
                  dt.datetime(2024, 3, 15, 17, 0)],
    "end":       [dt.datetime(2024, 3, 1, 11, 30),
                  dt.datetime(2024, 3, 7, 16, 0),
                  dt.datetime(2024, 3, 10, 12, 0),
                  dt.datetime(2024, 3, 16, 9, 30)],
})

events = events.with_columns([
    (pl.col("end") - pl.col("start")).alias("duration"),
    (pl.col("end") - pl.col("start")).dt.total_hours().alias("hours"),
    (pl.col("end") - pl.col("start")).dt.total_minutes().alias("minutes"),
    # Add 7 days to start
    (pl.col("start") + pl.duration(days=7)).alias("followup"),
])
print(events)
print(f"\nTotal event duration (sum of hours): {events['hours'].sum()} h")

# ──────────────────────────────────────────────────────────────
# 4. Parsing strings → Datetime with strptime
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. PARSING STRINGS → DATE / DATETIME")
print("=" * 60)

raw = pl.DataFrame({
    "date_str":     ["2024-01-15", "2024-02-28", "2024-03-10", "bad-date"],
    "datetime_str": ["2024-01-15 09:30:00", "2024-02-28 14:00:00",
                     "2024-03-10 00:01:00", "2024-04-01 12:00:00"],
    "eu_date_str":  ["15/01/2024", "28/02/2024", "10/03/2024", "01/04/2024"],
})

parsed = raw.with_columns([
    pl.col("date_str").str.to_date(format="%Y-%m-%d", strict=False).alias("date"),
    pl.col("datetime_str").str.to_datetime(format="%Y-%m-%d %H:%M:%S").alias("datetime"),
    pl.col("eu_date_str").str.to_date(format="%d/%m/%Y").alias("eu_date"),
])
print(parsed)
print("\ndtypes:", parsed.dtypes)

# ──────────────────────────────────────────────────────────────
# 5. .str namespace
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. .str NAMESPACE")
print("=" * 60)

df_str = pl.DataFrame({
    "full_name":  ["Alice Smith", "Bob Jones", "Carol White",
                   "  dave brown  ", "Eve"],
    "email":      ["alice@example.com", "bob@test.org", "carol@example.com",
                   "dave@corp.net", "INVALID"],
    "product_id": ["SKU-1001-A", "SKU-2034-B", "SKU-0099-C",
                   "SKU-4800-A", "SKU-3200-D"],
    "score_str":  ["88.5", "72.0", "95.3", "64.1", "n/a"],
})

result = df_str.with_columns([
    # length, case conversion, strip
    pl.col("full_name").str.len_chars().alias("name_len"),
    pl.col("full_name").str.to_uppercase().alias("upper_name"),
    pl.col("full_name").str.strip_chars().alias("clean_name"),

    # split → extract first name
    pl.col("full_name").str.strip_chars()
                       .str.split(" ")
                       .list.first()
                       .alias("first_name"),

    # contains / starts_with
    pl.col("email").str.contains("example.com").alias("is_example"),
    pl.col("email").str.starts_with("alice").alias("is_alice"),

    # replace
    pl.col("email").str.to_lowercase().alias("email_lower"),

    # regex extract: capture numeric part of SKU
    pl.col("product_id").str.extract(r"SKU-(\d+)-", group_index=1).alias("sku_num"),

    # cast score (null on failure)
    pl.col("score_str").cast(pl.Float64, strict=False).alias("score"),
])
print(result.select(["full_name", "clean_name", "first_name",
                     "name_len", "is_example", "sku_num", "score"]))

# ──────────────────────────────────────────────────────────────
# 6. Combined pipeline: parse log lines
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. COMBINED PIPELINE: PARSE LOG LINES")
print("=" * 60)

logs = pl.DataFrame({
    "raw_log": [
        "2024-03-01 08:12:34 INFO  user=alice action=login  duration_ms=53",
        "2024-03-01 09:45:00 WARN  user=bob   action=upload duration_ms=4821",
        "2024-03-01 10:00:01 ERROR user=carol  action=delete duration_ms=10",
        "2024-03-01 11:30:22 INFO  user=alice action=logout duration_ms=12",
        "2024-03-01 14:08:55 INFO  user=dave  action=login  duration_ms=88",
    ]
})

parsed_logs = logs.with_columns([
    # timestamp
    pl.col("raw_log").str.slice(0, 19)
      .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
      .alias("timestamp"),
    # level (word at position 20..24)
    pl.col("raw_log").str.extract(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (\w+)", group_index=1)
      .alias("level"),
    # user
    pl.col("raw_log").str.extract(r"user=(\w+)", group_index=1).alias("user"),
    # action
    pl.col("raw_log").str.extract(r"action=(\w+)", group_index=1).alias("action"),
    # duration
    pl.col("raw_log").str.extract(r"duration_ms=(\d+)", group_index=1)
      .cast(pl.Int64)
      .alias("duration_ms"),
])
print(parsed_logs.drop("raw_log"))

# Summary by user
summary = (
    parsed_logs
    .group_by("user")
    .agg([
        pl.len().alias("n_events"),
        pl.col("duration_ms").mean().round(1).alias("avg_ms"),
        pl.col("duration_ms").max().alias("max_ms"),
        pl.col("timestamp").min().alias("first_seen"),
        pl.col("timestamp").max().alias("last_seen"),
    ])
    .sort("avg_ms", descending=True)
)
print("\nUser summary:")
print(summary)

print("\n[Done] 05_datetime_strings.py completed successfully.")
