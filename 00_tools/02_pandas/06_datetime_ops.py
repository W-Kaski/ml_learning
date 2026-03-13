"""
02_pandas / 06_datetime_ops.py
================================
Topic: Working with dates, times, and time-series in pandas.

Covers:
  1. Parsing and creating datetime objects
  2. DatetimeIndex – setting and using as index
  3. dt accessor – extracting year, month, day, weekday, hour…
  4. Arithmetic: timedelta, shift, date_range
  5. Resampling (aggregate by frequency)
  6. Rolling and expanding windows
  7. Time zone handling
  8. Practical: daily sales KPIs with rolling metrics
"""

import numpy as np
import pandas as pd

np.random.seed(21)

# ──────────────────────────────────────────────────────────────
# 1. Parsing and creating datetimes
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. PARSING DATETIMES")
print("=" * 60)

# String parse
s = pd.Series(["2024-01-15", "2024/06/30", "15 Mar 2024", "2024-12-01 14:30:00"])
parsed = pd.to_datetime(s, format="mixed")
print("Parsed from mixed strings:")
print(parsed.to_string())
print(f"dtype: {parsed.dtype}")

# From components
dates_from_parts = pd.to_datetime({
    "year":  [2023, 2024, 2025],
    "month": [6,    1,    12],
    "day":   [15,   1,    31],
})
print("\nBuilt from year/month/day dict:")
print(dates_from_parts.tolist())

# pd.date_range
monthly = pd.date_range("2024-01-01", periods=12, freq="MS")
weekly  = pd.date_range("2024-01-01", "2024-03-31", freq="W-MON")
print(f"\nMonthly (12 periods): {monthly[0].date()} → {monthly[-1].date()}")
print(f"Weekly  Mondays    : {len(weekly)} dates, first={weekly[0].date()}, last={weekly[-1].date()}")

# ──────────────────────────────────────────────────────────────
# 2. DatetimeIndex
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATETIMEINDEX – INDEXING AND SLICING")
print("=" * 60)

dates = pd.date_range("2024-01-01", periods=365, freq="D")
ts = pd.Series(np.random.randn(365).cumsum() + 100, index=dates, name="price")

# Partial string indexing
print("March 2024 (first 5):")
print(ts["2024-03"].head(5).to_string())

print("\nQ2 2024 (April–June, last 5):")
print(ts["2024-04":"2024-06"].tail(5).to_string())

print(f"\nValue on 2024-07-04: {ts['2024-07-04']:.2f}")

# ──────────────────────────────────────────────────────────────
# 3. dt accessor
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. dt ACCESSOR – EXTRACT DATE COMPONENTS")
print("=" * 60)

df_events = pd.DataFrame({
    "event_time": pd.to_datetime([
        "2024-03-15 09:30:00", "2024-06-21 14:00:00",
        "2024-12-25 08:00:00", "2024-01-01 00:01:00",
        "2024-07-04 20:30:00",
    ]),
    "value": [10, 20, 30, 40, 50],
})

df_events["year"]     = df_events["event_time"].dt.year
df_events["month"]    = df_events["event_time"].dt.month
df_events["day"]      = df_events["event_time"].dt.day
df_events["hour"]     = df_events["event_time"].dt.hour
df_events["weekday"]  = df_events["event_time"].dt.day_name()
df_events["quarter"]  = df_events["event_time"].dt.quarter
df_events["week_num"] = df_events["event_time"].dt.isocalendar().week.astype(int)
df_events["is_weekend"] = df_events["event_time"].dt.dayofweek >= 5

print(df_events.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 4. Timedelta arithmetic
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TIMEDELTA ARITHMETIC")
print("=" * 60)

orders = pd.DataFrame({
    "order_id":    [1, 2, 3, 4, 5],
    "order_date":  pd.to_datetime(["2024-01-05", "2024-02-10", "2024-03-01",
                                   "2024-04-20", "2024-05-15"]),
    "ship_date":   pd.to_datetime(["2024-01-08", "2024-02-12", "2024-03-06",
                                   "2024-04-25", "2024-05-15"]),
})

orders["days_to_ship"] = (orders["ship_date"] - orders["order_date"]).dt.days
orders["on_time"]      = orders["days_to_ship"] <= 3
orders["due_date"]     = orders["order_date"] + pd.Timedelta(days=5)

print(orders.to_string(index=False))
print(f"\nAvg days to ship: {orders['days_to_ship'].mean():.1f}")
print(f"On-time rate    : {orders['on_time'].mean()*100:.0f}%")

# shift a time series
print("\nPrice series shifted by 7 days (lag-1 week):")
ts_lag = ts.shift(7)   # 7 trading-day lag
lag_df = pd.DataFrame({"price": ts["2024-02-01":"2024-02-07"],
                        "lag7":  ts_lag["2024-02-01":"2024-02-07"]}).round(2)
print(lag_df.to_string())

# ──────────────────────────────────────────────────────────────
# 5. Resampling
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. RESAMPLING")
print("=" * 60)

# Daily → weekly, monthly, quarterly
weekly_mean  = ts.resample("W").mean().round(2)
monthly_mean = ts.resample("ME").mean().round(2)
quarterly    = ts.resample("QE").agg(["mean", "min", "max"]).round(2)

print(f"Daily → Weekly  : {len(ts)} → {len(weekly_mean)} rows")
print(f"Daily → Monthly : {len(ts)} → {len(monthly_mean)} rows")
print("\nQuarterly OHLC-style stats:")
print(quarterly.to_string())

# Downsampling to business day (skip weekends — no change since data is daily)
bday = ts.resample("B").last()
print(f"\nBusiness-day resampled: {len(bday)} rows")

# Upsampling then forward-fill
monthly_idx = ts.resample("MS").first()
daily_upsample = monthly_idx.resample("D").ffill()
print(f"\nMonthly → Daily via ffill: {len(monthly_idx)} → {len(daily_upsample)} rows")

# ──────────────────────────────────────────────────────────────
# 6. Rolling and expanding windows
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ROLLING AND EXPANDING WINDOWS")
print("=" * 60)

roll7   = ts.rolling(window=7).mean()
roll30  = ts.rolling(window=30).mean()
roll30_std = ts.rolling(window=30).std()
expanding_mean = ts.expanding().mean()

sample_df = pd.DataFrame({
    "price":     ts,
    "MA7":       roll7,
    "MA30":      roll30,
    "std30":     roll30_std,
    "cum_mean":  expanding_mean,
}).dropna().round(2)

print("Rolling stats sample (Jan 31 – Feb 5):")
print(sample_df.loc["2024-01-31":"2024-02-05"].to_string())

# Bollinger-band style bounds
sample_df["upper_band"] = (sample_df["MA30"] + 2 * sample_df["std30"]).round(2)
sample_df["lower_band"] = (sample_df["MA30"] - 2 * sample_df["std30"]).round(2)
above = (sample_df["price"] > sample_df["upper_band"]).sum()
below = (sample_df["price"] < sample_df["lower_band"]).sum()
print(f"\nValues above upper band (2σ): {above}")
print(f"Values below lower band (2σ): {below}")

# ──────────────────────────────────────────────────────────────
# 7. Time zone handling
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TIME ZONE HANDLING")
print("=" * 60)

dt_utc = pd.Timestamp("2024-06-15 12:00:00", tz="UTC")
dt_ny  = dt_utc.tz_convert("America/New_York")
dt_tok = dt_utc.tz_convert("Asia/Tokyo")
dt_lon = dt_utc.tz_convert("Europe/London")

print(f"UTC    : {dt_utc}")
print(f"New York: {dt_ny}")
print(f"Tokyo  : {dt_tok}")
print(f"London : {dt_lon}")

# Localize a naive Series then convert
naive_series = pd.date_range("2024-01-01", periods=4, freq="6h")
localized  = naive_series.tz_localize("UTC")
converted  = localized.tz_convert("US/Pacific")
tz_df = pd.DataFrame({"UTC": naive_series, "US_Pacific": converted})
print("\nNaive UTC → US/Pacific:")
print(tz_df.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 8. Practical: daily sales with rolling KPIs
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. PRACTICAL – daily sales KPI dashboard")
print("=" * 60)

sales_dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
sales = pd.Series(
    np.random.poisson(lam=5000, size=len(sales_dates))
    + np.sin(np.linspace(0, 4 * np.pi, len(sales_dates))) * 1000,
    index=sales_dates,
    name="daily_sales",
).round(0)

kpi = pd.DataFrame({
    "sales":        sales,
    "MA7":          sales.rolling(7).mean().round(0),
    "MA30":         sales.rolling(30).mean().round(0),
    "YTD_total":    sales.expanding().sum().round(0),
    "pct_chg_7d":   sales.pct_change(7).mul(100).round(2),
})

print("Last 7 days of the year:")
print(kpi.tail(7).to_string())

# Month-over-month growth
monthly_sales = sales.resample("ME").sum()
mom_growth    = monthly_sales.pct_change().mul(100).round(2)
print("\nMonthly sales & MoM growth:")
summary = pd.DataFrame({"sales": monthly_sales, "MoM_%": mom_growth})
summary.index = summary.index.strftime("%b")
print(summary.to_string())

print("\n[Done] 06_datetime_ops.py completed successfully.")
