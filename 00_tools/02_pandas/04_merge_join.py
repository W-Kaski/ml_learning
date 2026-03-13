"""
02_pandas / 04_merge_join.py
==============================
Topic: Combining DataFrames with merge, join, and concat.

Covers:
  1. Building relational tables (orders / customers / products)
  2. Inner, left, right, outer merge (how=)
  3. Merge on multiple keys
  4. Indicator column (merge with _merge flag)
  5. Self-join (e.g., employee → manager)
  6. pd.concat – stacking rows and columns
  7. Avoiding common pitfalls: duplicate keys, many-to-many explosion
  8. Combining sources: enrich orders with customer + product info
"""

import numpy as np
import pandas as pd

np.random.seed(3)

# ──────────────────────────────────────────────────────────────
# 1. Build relational tables
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. BUILD RELATIONAL TABLES")
print("=" * 60)

customers = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 105],
    "name":        ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "city":        ["NYC", "LA", "NYC", "Chicago", "NYC"],
    "tier":        ["Gold", "Silver", "Bronze", "Gold", "Silver"],
})

products = pd.DataFrame({
    "product_id":  [201, 202, 203, 204],
    "product_name":["Laptop", "Phone", "Tablet", "Watch"],
    "category":    ["Electronics", "Electronics", "Electronics", "Accessories"],
    "price":       [999.99, 599.99, 399.99, 199.99],
})

orders = pd.DataFrame({
    "order_id":   range(1001, 1021),
    "customer_id": np.random.choice([101, 102, 103, 104, 105, 106], 20),  # 106 = unknown
    "product_id":  np.random.choice([201, 202, 203, 204, 205], 20),       # 205 = discontinued
    "quantity":    np.random.randint(1, 5, 20),
    "order_date":  pd.date_range("2024-01-01", periods=20, freq="3D"),
})

print("customers:\n", customers.to_string(index=False))
print("\nproducts:\n", products.to_string(index=False))
print("\norders (head 6):\n", orders.head(6).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 2. Inner, left, right, outer merge
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. INNER / LEFT / RIGHT / OUTER MERGE")
print("=" * 60)

inner = pd.merge(orders, customers, on="customer_id", how="inner")
left  = pd.merge(orders, customers, on="customer_id", how="left")
# right join: all customers, even those with no orders
right = pd.merge(orders, customers, on="customer_id", how="right")
outer = pd.merge(orders, customers, on="customer_id", how="outer")

print(f"orders rows       : {len(orders)}")
print(f"inner join rows   : {len(inner):3d}  (only orders with known customer)")
print(f"left  join rows   : {len(left):3d}  (all orders; NaN for unknown customers)")
print(f"right join rows   : {len(right):3d}  (all customers incl. those without orders)")
print(f"outer join rows   : {len(outer):3d}  (everything)")

# Show the unknown customer rows in left join
unknown_cust = left[left["name"].isna()][["order_id", "customer_id", "name"]]
print(f"\nOrders with unknown customers ({len(unknown_cust)}):")
print(unknown_cust.to_string(index=False))

# Customers with no orders (right join)
no_orders = right[right["order_id"].isna()][["customer_id", "name"]]
print(f"\nCustomers with no orders:")
print(no_orders.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 3. Merge on multiple keys
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MULTI-KEY MERGE – stock table (product × warehouse)")
print("=" * 60)

stock = pd.DataFrame({
    "product_id":   [201, 201, 202, 202, 203],
    "warehouse":    ["East", "West", "East", "West", "East"],
    "stock_qty":    [ 100,    50,    200,    80,    300],
})
reorder = pd.DataFrame({
    "product_id": [201, 202, 203],
    "warehouse":  ["East", "West", "East"],
    "reorder_pt": [20,    30,    50],
})
stock_full = pd.merge(stock, reorder, on=["product_id", "warehouse"], how="left")
stock_full["needs_reorder"] = (
    stock_full["stock_qty"] < stock_full["reorder_pt"].fillna(0)
)
print(stock_full.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 4. Indicator column (_merge)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. INDICATOR COLUMN (indicator=True)")
print("=" * 60)

orders_with_flag = pd.merge(
    orders[["order_id", "customer_id"]],
    customers[["customer_id", "name"]],
    on="customer_id", how="outer", indicator=True
)
print(orders_with_flag["_merge"].value_counts().to_string())

# ──────────────────────────────────────────────────────────────
# 5. Self-join: employee → manager
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SELF-JOIN – employee → manager name")
print("=" * 60)

employees = pd.DataFrame({
    "emp_id":  [1, 2, 3, 4, 5, 6],
    "name":    ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
    "mgr_id":  [None, 1, 1, 2, 2, 3],   # Alice is CEO (no manager)
})
emp_mgr = pd.merge(
    employees,
    employees[["emp_id", "name"]].rename(columns={"emp_id": "mgr_id", "name": "manager_name"}),
    on="mgr_id", how="left"
)
print(emp_mgr[["emp_id", "name", "mgr_id", "manager_name"]].to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 6. pd.concat – stacking rows and columns
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. pd.concat – row stack and column stack")
print("=" * 60)

q1 = pd.DataFrame({"month": [1, 2, 3], "revenue": [10000, 12000, 11500]})
q2 = pd.DataFrame({"month": [4, 5, 6], "revenue": [13000, 14500, 15000]})

# Row concatenation
half_year = pd.concat([q1, q2], ignore_index=True)
print("Row concat (H1 revenue):")
print(half_year.to_string(index=False))

# Column concatenation
extra = pd.DataFrame({"cost": [8000, 9000, 9500, 10000, 11000, 12000]})
full  = pd.concat([half_year, extra], axis=1)
full["profit"] = full["revenue"] - full["cost"]
print("\nColumn concat (+ cost & profit):")
print(full.to_string(index=False))

# keys parameter to track source
labeled = pd.concat([q1, q2], keys=["Q1", "Q2"])
print(f"\nConcat with keys – index levels: {labeled.index.names}")
print(labeled.to_string())

# ──────────────────────────────────────────────────────────────
# 7. Many-to-many pitfall
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. MANY-TO-MANY EXPLOSION WARNING")
print("=" * 60)

# Simulate duplicate keys on both sides
a = pd.DataFrame({"key": ["x", "x", "y"], "val_a": [1, 2, 3]})
b = pd.DataFrame({"key": ["x", "x", "z"], "val_b": [10, 20, 30]})
m2m = pd.merge(a, b, on="key", how="inner")
print(f"Left rows: {len(a)}, Right rows: {len(b)}, Merged rows: {len(m2m)}")
print("(key='x' produced 2×2=4 rows — many-to-many!)")
print(m2m.to_string(index=False))
print("\nFix: deduplicate before merge, or use validate='1:1'/'1:m'/'m:1'")
try:
    pd.merge(a, b, on="key", how="inner", validate="1:1")
except pd.errors.MergeError as e:
    print(f"MergeError caught: {e}")

# ──────────────────────────────────────────────────────────────
# 8. Full enrichment: orders → customers + products
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. FULL ENRICHMENT – orders with customer + product info")
print("=" * 60)

enriched = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(products,  on="product_id",  how="left")
)
enriched["line_total"] = (enriched["quantity"] * enriched["price"]).round(2)
enriched["price"] = enriched["price"].fillna(0)
enriched["line_total"] = enriched["line_total"].fillna(0)

print(enriched[["order_id", "name", "product_name", "quantity", "price", "line_total"]]
      .head(8).to_string(index=False))

# Revenue by customer tier
rev_by_tier = (
    enriched.groupby("tier")["line_total"]
    .sum().round(2).sort_values(ascending=False)
)
print("\nRevenue by tier:")
print(rev_by_tier.to_string())

print("\n[Done] 04_merge_join.py completed successfully.")
