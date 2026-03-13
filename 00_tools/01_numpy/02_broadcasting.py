#!/usr/bin/env python3
"""
02_broadcasting.py

NumPy 第三课：广播机制（Broadcasting）。

运行：
python3 02_broadcasting.py
"""

import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def explain_shape(name: str, arr: np.ndarray):
    print(f"{name}.shape = {arr.shape}")
    print(arr)


def main():
    np.random.seed(42)

    section("1) 广播直观例子：矩阵 + 标量")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 10
    c = a + b
    explain_shape("a", a)
    print("b =", b)
    explain_shape("c = a + b", c)

    section("2) 矩阵 + 行向量")
    m = np.array([[1, 2, 3], [4, 5, 6]])
    row = np.array([100, 200, 300])  # shape (3,)
    out_row = m + row
    explain_shape("m", m)
    explain_shape("row", row)
    explain_shape("m + row", out_row)

    section("3) 矩阵 + 列向量")
    col = np.array([[10], [20]])  # shape (2,1)
    out_col = m + col
    explain_shape("col", col)
    explain_shape("m + col", out_col)

    section("4) 广播规则（从后往前对齐）")
    print("规则：两个维度兼容，当且仅当：")
    print("1) 两维相等，或")
    print("2) 其中一维是 1")
    print("否则报错")

    x = np.ones((2, 3, 4))
    y = np.ones((3, 1))
    z = x + y
    print("例子: (2,3,4) + (3,1) ->", z.shape)

    section("5) 常见报错演示与修复")
    u = np.ones((2, 3))
    v = np.array([1, 2])  # shape (2,) 与 (2,3) 不兼容
    print("u.shape:", u.shape, "v.shape:", v.shape)
    try:
        _ = u + v
    except ValueError as e:
        print("报错:", e)

    print("修复方法1: 让 v 变成列向量 (2,1)")
    v_col = v.reshape(2, 1)
    fixed1 = u + v_col
    explain_shape("v_col", v_col)
    explain_shape("u + v_col", fixed1)

    print("修复方法2: 如果你本意是按列加，改成长度为3的行向量")
    v_row = np.array([1, 2, 3])
    fixed2 = u + v_row
    explain_shape("v_row", v_row)
    explain_shape("u + v_row", fixed2)

    section("6) 实战小例子：批量标准化")
    # 假设 batch 数据 shape=(batch, feature)
    data = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ]
    )
    mean = data.mean(axis=0)  # shape (3,)
    std = data.std(axis=0)    # shape (3,)
    normalized = (data - mean) / (std + 1e-8)

    explain_shape("data", data)
    print("mean:", mean)
    print("std:", std)
    explain_shape("normalized", normalized)
    print("normalized 每列均值(约等于0):", normalized.mean(axis=0))

    section("学习总结")
    print("1. 广播是 NumPy 向量化的核心之一。")
    print("2. 先看 shape，再从后向前对齐维度判断是否兼容。")
    print("3. reshape/expand 维度是修复广播报错最常用的方法。")


if __name__ == "__main__":
    main()
