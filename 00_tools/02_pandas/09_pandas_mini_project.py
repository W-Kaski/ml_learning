"""
02_pandas / 09_pandas_mini_project.py
========================================
Mini-project: End-to-End Customer Sales Analysis Pipeline

Scenario:
  An e-commerce company has three data sources:
    - customers.csv  (customer profile)
    - orders.csv     (order transactions)
    - products.csv   (product catalogue)

  Goal: build a clean, enriched analysis table and produce
  a business-readable KPI report.

Pipeline stages:
  1.  Generate and save raw data files
  2.  Load & validate (schema, dtypes, nulls)
  3.  Clean data (missing, outliers, type fixes)
  4.  Merge sources into one enriched table
  5.  Feature engineering (CLV, recency, order_value_tier …)
  6.  Aggregated KPI summary (by segment, by month, by product)
  7.  Cohort analysis: monthly retention hint
  8.  Save final report to CSV + print summary
"""

import json
import os
import numpy as np
import pandas as pd

np.random.seed(2024)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "mini_project")
os.makedirs(DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# STAGE 1 – Generate raw CSVs
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STAGE 1 – GENERATE RAW DATA")
print("=" * 60)

N_CUST = 500
N_PROD = 80
N_ORD  = 3000

# customers
customers = pd.DataFrame({
    "customer_id": range(1, N_CUST + 1),
    "name":        [f"Customer_{i}" for i in range(1, N_CUST + 1)],
    "email":       [f"c{i}@mail.com" for i in range(1, N_CUST + 1)],
    "country":     np.random.choice(["US", "UK", "DE", "FR", "CA"],
                                    N_CUST, p=[0.40, 0.20, 0.15, 0.15, 0.10]),
    "segment":     np.random.choice(["Consumer", "Corporate", "Home Office"],
                                    N_CUST, p=[0.52, 0.32, 0.16]),
    "join_date":   pd.to_datetime(
        np.random.choice(pd.date_range("2020-01-01", "2022-12-31"), N_CUST)
    ),
})
# Inject 3% missing country
customers.loc[np.random.choice(N_CUST, 15, replace=False), "country"] = np.nan

# products
categories = np.random.choice(
    ["Electronics", "Clothing", "Books", "Home", "Sports"], N_PROD,
    p=[0.25, 0.20, 0.20, 0.20, 0.15])
products = pd.DataFrame({
    "product_id":   range(1001, 1001 + N_PROD),
    "product_name": [f"Product_{i}" for i in range(N_PROD)],
    "category":     categories,
    "sub_category": [f"Sub_{c[:3]}_{i%5}" for i, c in enumerate(categories)],
    "cost_price":   np.random.uniform(5, 200, N_PROD).round(2),
    "sell_price":   np.random.uniform(10, 400, N_PROD).round(2),
})
products["margin"] = ((products["sell_price"] - products["cost_price"]) /
                      products["sell_price"] * 100).round(2)

# orders  (2022-2024)
all_dates = pd.date_range("2022-01-01", "2024-12-31")
orders = pd.DataFrame({
    "order_id":    range(10001, 10001 + N_ORD),
    "customer_id": np.random.choice(range(1, N_CUST + 1), N_ORD),
    "product_id":  np.random.choice(range(1001, 1001 + N_PROD), N_ORD),
    "order_date":  pd.to_datetime(np.random.choice(all_dates, N_ORD)),
    "quantity":    np.random.randint(1, 10, N_ORD),
    "discount":    np.random.choice([0, 0.05, 0.10, 0.15, 0.20], N_ORD,
                                    p=[0.50, 0.20, 0.15, 0.10, 0.05]),
    "ship_days":   np.random.randint(1, 8, N_ORD),
})
# Inject some bad data (negative quantity, future date)
orders.loc[np.random.choice(N_ORD, 20, replace=False), "quantity"] = -1
orders.loc[np.random.choice(N_ORD, 5,  replace=False), "order_date"] = pd.Timestamp("2026-06-01")

# Save
customers.to_csv(f"{DATA_DIR}/customers.csv", index=False)
products.to_csv(f"{DATA_DIR}/products.csv",   index=False)
orders.to_csv(f"{DATA_DIR}/orders.csv",       index=False)
print(f"Saved: customers ({len(customers)}), "
      f"products ({len(products)}), orders ({len(orders)})")

# ══════════════════════════════════════════════════════════════
# STAGE 2 – Load & Validate
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 2 – LOAD & VALIDATE")
print("=" * 60)

customers = pd.read_csv(f"{DATA_DIR}/customers.csv", parse_dates=["join_date"])
products  = pd.read_csv(f"{DATA_DIR}/products.csv")
orders    = pd.read_csv(f"{DATA_DIR}/orders.csv",    parse_dates=["order_date"])

def validate_table(name, df):
    print(f"\n[{name}]  shape={df.shape}")
    dup = df.duplicated().sum()
    missing = df.isna().sum()[df.isna().sum() > 0]
    print(f"  Duplicates: {dup}")
    if len(missing):
        print(f"  Missing:\n{missing.to_string()}")
    else:
        print("  Missing: none")
    print(f"  dtypes: { {c: str(t) for c, t in df.dtypes.items()} }")

validate_table("customers", customers)
validate_table("products",  products)
validate_table("orders",    orders)

# ══════════════════════════════════════════════════════════════
# STAGE 3 – Clean
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 3 – CLEAN")
print("=" * 60)

# 3a. Customers: fill missing country with mode
mode_country = customers["country"].mode()[0]
customers["country"] = customers["country"].fillna(mode_country)
print(f"Filled {customers['country'].isna().sum()} country NAs with '{mode_country}'")

# 3b. Orders: remove invalid (-1 quantity) and future orders
ref_date = pd.Timestamp("2025-01-01")
bad_qty  = orders["quantity"] <= 0
bad_date = orders["order_date"] > ref_date
print(f"Removing {bad_qty.sum()} negative-qty orders")
print(f"Removing {bad_date.sum()} future-date orders")
orders = orders[~bad_qty & ~bad_date].copy()
print(f"Orders after clean: {len(orders)}")

# 3c. Products: clip cost/sell price to positive
products["cost_price"] = products["cost_price"].clip(lower=0.01)
products["sell_price"] = products["sell_price"].clip(lower=products["cost_price"])

# ══════════════════════════════════════════════════════════════
# STAGE 4 – Merge
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 4 – MERGE")
print("=" * 60)

enriched = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(products[["product_id", "product_name", "category",
                      "sell_price", "cost_price", "margin"]],
           on="product_id", how="left")
)

# Derived columns
enriched["gross_revenue"] = (enriched["sell_price"] * enriched["quantity"]
                              * (1 - enriched["discount"])).round(2)
enriched["cogs"]          = (enriched["cost_price"] * enriched["quantity"]).round(2)
enriched["gross_profit"]  = (enriched["gross_revenue"] - enriched["cogs"]).round(2)
enriched["order_month"]   = enriched["order_date"].dt.to_period("M")

print(f"Enriched table shape: {enriched.shape}")
print(f"Missing after merge: {enriched.isna().sum()[enriched.isna().sum()>0].to_string()}")

# ══════════════════════════════════════════════════════════════
# STAGE 5 – Feature engineering
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 5 – FEATURE ENGINEERING")
print("=" * 60)

analysis_date = pd.Timestamp("2025-01-01")

# Per-customer aggregates
customer_stats = (
    enriched.groupby("customer_id")
    .agg(
        total_orders    = ("order_id",       "nunique"),
        total_revenue   = ("gross_revenue",  "sum"),
        total_profit    = ("gross_profit",   "sum"),
        avg_order_value = ("gross_revenue",  "mean"),
        last_order      = ("order_date",     "max"),
        first_order     = ("order_date",     "min"),
    )
    .reset_index()
)
customer_stats["recency_days"] = (analysis_date - customer_stats["last_order"]).dt.days
customer_stats["tenure_days"]  = (customer_stats["last_order"] - customer_stats["first_order"]).dt.days

# CLV proxy: total revenue
customer_stats["clv_tier"] = pd.qcut(
    customer_stats["total_revenue"],
    q=4, labels=["Low", "Mid", "High", "VIP"]
)

print("Customer stats (first 5):")
print(customer_stats[[
    "customer_id", "total_orders", "total_revenue",
    "recency_days", "tenure_days", "clv_tier"
]].head(5).to_string(index=False))

# Order value tier
enriched["order_value_tier"] = pd.qcut(
    enriched["gross_revenue"], q=3, labels=["Small", "Medium", "Large"]
)

# ══════════════════════════════════════════════════════════════
# STAGE 6 – KPI Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 6 – KPI SUMMARY")
print("=" * 60)

total_rev    = enriched["gross_revenue"].sum()
total_profit = enriched["gross_profit"].sum()
total_orders = enriched["order_id"].nunique()
unique_cust  = enriched["customer_id"].nunique()

print(f"Total Revenue  : ${total_rev:,.0f}")
print(f"Total Profit   : ${total_profit:,.0f}  ({total_profit/total_rev*100:.1f}% margin)")
print(f"Total Orders   : {total_orders:,}")
print(f"Unique Customers: {unique_cust:,}")
print(f"Avg Order Value : ${total_rev/total_orders:,.2f}")

# By segment
print("\nRevenue by customer segment:")
seg = enriched.merge(customers[["customer_id", "segment"]], on="customer_id", how="left",
                     suffixes=("", "_cust"))
seg_kpi = seg.groupby("segment_cust").agg(
    revenue = ("gross_revenue", "sum"),
    profit  = ("gross_profit",  "sum"),
    orders  = ("order_id",      "nunique"),
).round(2)
seg_kpi["margin%"] = (seg_kpi["profit"] / seg_kpi["revenue"] * 100).round(2)
print(seg_kpi.to_string())

# By category
print("\nRevenue by product category:")
cat_kpi = enriched.groupby("category").agg(
    revenue = ("gross_revenue", "sum"),
    profit  = ("gross_profit",  "sum"),
    units   = ("quantity",      "sum"),
).sort_values("revenue", ascending=False).round(2)
cat_kpi["margin%"] = (cat_kpi["profit"] / cat_kpi["revenue"] * 100).round(2)
print(cat_kpi.to_string())

# Monthly trend
print("\nMonthly revenue trend  (last 6 months of 2024):")
monthly = (
    enriched[enriched["order_date"] >= "2024-07-01"]
    .groupby("order_month")["gross_revenue"].sum()
    .round(0)
)
monthly.index = monthly.index.astype(str)
print(monthly.to_string())

# ══════════════════════════════════════════════════════════════
# STAGE 7 – Cohort analysis (simplified)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 7 – COHORT ANALYSIS (2022 join cohort)")
print("=" * 60)

# Cohort = month of first purchase
first_purchase = (
    enriched.groupby("customer_id")["order_date"].min()
    .dt.to_period("M").rename("cohort")
)
enriched = enriched.merge(first_purchase.reset_index(), on="customer_id", how="left")
enriched["cohort_age"] = (
    enriched["order_month"].astype("int64") - enriched["cohort"].astype("int64")
)

cohort_counts = (
    enriched[enriched["cohort"].astype(str).str.startswith("2022")]
    .groupby(["cohort", "cohort_age"])["customer_id"].nunique()
    .unstack(fill_value=0)
    .iloc[:, :6]  # first 6 months of retention
)
cohort_counts.index = cohort_counts.index.astype(str)
cohort_counts.columns = [f"Month+{c}" for c in cohort_counts.columns]

# Normalize by cohort size (month 0)
retention = cohort_counts.div(cohort_counts["Month+0"], axis=0).round(3) * 100
print("Retention % (2022 cohorts, first 6 months):")
print(cohort_counts.head(6).to_string())
print("\nRetention rate %:")
print(retention.head(6).to_string())

# ══════════════════════════════════════════════════════════════
# STAGE 8 – Save outputs
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 8 – SAVE OUTPUTS")
print("=" * 60)

output_dir = os.path.join(DATA_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

# Enriched table
enriched.drop(columns=["order_month", "cohort", "cohort_age"],
              errors="ignore").to_csv(f"{output_dir}/enriched_orders.csv", index=False)

# Customer stats
customer_stats.to_csv(f"{output_dir}/customer_kpis.csv", index=False)

# Category KPI
cat_kpi.reset_index().to_csv(f"{output_dir}/category_kpis.csv", index=False)

# Summary metrics as JSON
summary = {
    "total_revenue":  round(float(total_rev), 2),
    "total_profit":   round(float(total_profit), 2),
    "margin_pct":     round(float(total_profit / total_rev * 100), 2),
    "total_orders":   int(total_orders),
    "unique_customers": int(unique_cust),
    "avg_order_value":round(float(total_rev / total_orders), 2),
}
with open(f"{output_dir}/summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved to: {output_dir}/")
print(f"  enriched_orders.csv    ({len(enriched):,} rows)")
print(f"  customer_kpis.csv      ({len(customer_stats):,} rows)")
print(f"  category_kpis.csv      ({len(cat_kpi):,} rows)")
print(f"  summary_metrics.json")
print(f"\nSummary metrics:")
print(json.dumps(summary, indent=2))

print("\n[Done] 09_pandas_mini_project.py completed successfully.")
