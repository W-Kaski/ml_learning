"""
03_polars / 04_join_concat.py
================================
Topic: Combining DataFrames in Polars – joins and concatenation.

Covers:
  1. inner / left / right / full (outer) join
  2. semi join and anti join
  3. Cross join
  4. Join on multiple keys
  5. Asof join (nearest-key time-series join)
  6. pl.concat – vertical and horizontal stacking
  7. Diagonal concat (union of different schemas)
  8. Lazy joins
"""

import polars as pl
import numpy as np
import datetime as dt

np.random.seed(8)

# ──────────────────────────────────────────────────────────────
# Build tables
# ──────────────────────────────────────────────────────────────
customers = pl.DataFrame({
    "customer_id": [101, 102, 103, 104, 105],
    "name":        ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "country":     ["US", "UK", "US", "DE", "US"],
    "tier":        ["Gold", "Silver", "Bronze", "Gold", "Silver"],
})

products = pl.DataFrame({
    "product_id":   [201, 202, 203, 204],
    "product_name": ["Laptop", "Phone", "Tablet", "Watch"],
    "category":     ["Electronics", "Electronics", "Electronics", "Accessories"],
    "price":        [999.99, 599.99, 399.99, 199.99],
})

# Orders – include unknown customer (106) and discontinued product (205)
np.random.seed(8)
orders = pl.DataFrame({
    "order_id":    list(range(1001, 1021)),
    "customer_id": np.random.choice([101, 102, 103, 104, 105, 106], 20).tolist(),
    "product_id":  np.random.choice([201, 202, 203, 204, 205], 20).tolist(),
    "quantity":    np.random.randint(1, 5, 20).tolist(),
    "order_date":  pl.date_range(
        pl.date(2024, 1, 1), pl.date(2024, 4, 9),
        interval="5d", eager=True
    ).to_list(),
})

print("customers:\n", customers)
print("\nproducts:\n", products)
print("\norders (head 6):\n", orders.head(6))

# ──────────────────────────────────────────────────────────────
# 1. inner / left / right / full join
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. INNER / LEFT / RIGHT / FULL JOIN")
print("=" * 60)

inner = orders.join(customers, on="customer_id", how="inner")
left  = orders.join(customers, on="customer_id", how="left")
full  = orders.join(customers, on="customer_id", how="full",
                    coalesce=True)

print(f"orders rows : {orders.height}")
print(f"inner rows  : {inner.height}  (only known customers)")
print(f"left  rows  : {left.height}   (all orders, null for unknown customer)")
print(f"full  rows  : {full.height}   (everything)")

# Show unknown customer rows
unknown = left.filter(pl.col("name").is_null())
print(f"\nOrders with unknown customer ({unknown.height}):")
print(unknown.select(["order_id", "customer_id", "name"]))

# ──────────────────────────────────────────────────────────────
# 2. Semi and anti join
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. SEMI JOIN and ANTI JOIN")
print("=" * 60)

# Semi: keep orders that have a matching customer (no customer columns added)
semi = orders.join(customers, on="customer_id", how="semi")
print(f"Semi join rows: {semi.height}  (orders with known customer, no extra cols)")
print(f"Semi columns: {semi.columns}")

# Anti: orders with NO matching customer
anti = orders.join(customers, on="customer_id", how="anti")
print(f"\nAnti join rows: {anti.height}  (orders with unknown customer)")
print(anti.select(["order_id", "customer_id"]))

# ──────────────────────────────────────────────────────────────
# 3. Cross join
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CROSS JOIN  (Cartesian product)")
print("=" * 60)

colors = pl.DataFrame({"color": ["Red", "Blue", "Green"]})
sizes  = pl.DataFrame({"size":  ["S", "M", "L", "XL"]})
variants = colors.join(sizes, how="cross")
print(f"Cross join: {len(colors)} colors × {len(sizes)} sizes = {variants.height} rows")
print(variants)

# ──────────────────────────────────────────────────────────────
# 4. Multi-key join
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. MULTI-KEY JOIN")
print("=" * 60)

stock = pl.DataFrame({
    "product_id": [201, 201, 202, 202, 203],
    "warehouse":  ["East", "West", "East", "West", "East"],
    "stock_qty":  [100, 50, 200, 80, 300],
})
reorder = pl.DataFrame({
    "product_id": [201, 202, 203],
    "warehouse":  ["East", "West", "East"],
    "reorder_pt": [20, 30, 50],
})
joined_multi = stock.join(reorder, on=["product_id", "warehouse"], how="left")
print(joined_multi)

# ──────────────────────────────────────────────────────────────
# 5. Asof join (nearest time-series join)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ASOF JOIN  (nearest price lookup)")
print("=" * 60)

prices = pl.DataFrame({
    "date":  [dt.date(2024, 1, 1), dt.date(2024, 1, 5),
              dt.date(2024, 1, 10), dt.date(2024, 1, 20)],
    "price": [100.0, 102.5, 101.0, 105.0],
}).sort("date")

trades = pl.DataFrame({
    "trade_date": [dt.date(2024, 1, 3), dt.date(2024, 1, 8),
                   dt.date(2024, 1, 15), dt.date(2024, 1, 22)],
    "qty":        [10, 5, 20, 8],
}).sort("trade_date")

# For each trade, find the latest price on or before trade_date
asof = trades.join_asof(prices, left_on="trade_date", right_on="date", strategy="backward")
asof = asof.with_columns((pl.col("price") * pl.col("qty")).alias("trade_value"))
print("Asof join: trade gets the most recent price:")
print(asof)

# ──────────────────────────────────────────────────────────────
# 6. pl.concat – vertical and horizontal
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. pl.concat – VERTICAL AND HORIZONTAL")
print("=" * 60)

q1 = pl.DataFrame({"month": [1, 2, 3], "revenue": [10000, 12000, 11500]})
q2 = pl.DataFrame({"month": [4, 5, 6], "revenue": [13000, 14500, 15000]})

# Vertical (stack rows)
half_year = pl.concat([q1, q2])
print("Vertical concat (H1):")
print(half_year)

# Horizontal (stack columns)
cost = pl.DataFrame({"cost": [8000, 9000, 9500, 10000, 11000, 12000]})
full  = pl.concat([half_year, cost], how="horizontal")
print("\nHorizontal concat (+ cost col):")
print(full)

# rechunk: consolidate Arrow chunks for better performance
rechunked = pl.concat([q1, q2], rechunk=True)
print(f"\nRechunked chunks: {rechunked['month'].n_chunks()}")

# ──────────────────────────────────────────────────────────────
# 7. Diagonal concat (different schemas, fills missing cols with null)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. DIAGONAL CONCAT (union of different schemas)")
print("=" * 60)

batch_a = pl.DataFrame({"id": [1, 2], "value": [10, 20], "source": ["A", "A"]})
batch_b = pl.DataFrame({"id": [3, 4], "value": [30, 40], "extra_col": [99, 100]})

diagonal = pl.concat([batch_a, batch_b], how="diagonal")
print("Diagonal concat (null for missing cols):")
print(diagonal)

# ──────────────────────────────────────────────────────────────
# 8. Lazy joins
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. LAZY JOINS")
print("=" * 60)

import os
csv_path = os.path.join(os.path.dirname(__file__), "data", "employees_large.csv")

if os.path.exists(csv_path):
    dept_budget = pl.LazyFrame({
        "dept":   ["Eng", "HR", "Sales", "Finance", "Legal"],
        "budget": [5_000_000, 2_000_000, 3_000_000, 2_500_000, 1_500_000],
    })

    result = (
        pl.scan_csv(csv_path)
        .join(dept_budget, on="dept", how="left")
        .with_columns(
            (pl.col("salary") / pl.col("budget") * 100).round(4).alias("salary_pct_budget")
        )
        .group_by("dept")
        .agg([
            pl.col("salary").mean().round(0).alias("avg_salary"),
            pl.col("salary_pct_budget").mean().round(4).alias("avg_salary_pct"),
            pl.col("budget").first().alias("dept_budget"),
            pl.len().alias("n"),
        ])
        .sort("avg_salary", descending=True)
        .collect()
    )
    print("Lazy join result: salary vs dept budget")
    print(result)
else:
    print("(skipped – run 02_lazy_api.py first to generate the CSV)")

print("\n[Done] 04_join_concat.py completed successfully.")
