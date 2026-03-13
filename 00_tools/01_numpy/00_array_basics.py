#!/usr/bin/env python3
"""
00_array_basics.py

NumPy 第一课：数组创建、形状、数据类型、变形、基础统计。

运行：
python3 00_array_basics.py
"""

import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_array(name: str, arr: np.ndarray):
    print(f"{name}:")
    print(arr)
    print(f"  shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}")


def main():
    np.random.seed(42)

    section("1) 创建 ndarray")
    a = np.array([1, 2, 3, 4])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = np.zeros((2, 3))
    d = np.ones((2, 3), dtype=np.int32)
    e = np.arange(0, 12, 2)
    f = np.linspace(0, 1, 5)

    show_array("a", a)
    show_array("b", b)
    show_array("c", c)
    show_array("d", d)
    show_array("e", e)
    show_array("f", f)

    section("2) 数据类型与类型转换")
    x = np.array([1.2, 3.4, 5.6])
    show_array("x", x)
    x_int = x.astype(np.int32)
    show_array("x_int", x_int)

    section("3) 形状操作：reshape / flatten / transpose")
    m = np.arange(1, 13).reshape(3, 4)
    show_array("m", m)

    m_flat = m.flatten()
    show_array("m_flat", m_flat)

    m_t = m.T
    show_array("m_t", m_t)

    m_2_2_3 = m.reshape(2, 2, 3)
    show_array("m_2_2_3", m_2_2_3)

    section("4) 基础运算")
    p = np.array([1, 2, 3])
    q = np.array([10, 20, 30])
    print(f"p + q = {p + q}")
    print(f"p - q = {p - q}")
    print(f"p * q = {p * q}")
    print(f"q / p = {q / p}")
    print(f"p ** 2 = {p ** 2}")

    section("5) 聚合统计")
    stats_arr = np.array([[1, 2, 3], [4, 5, 6]])
    show_array("stats_arr", stats_arr)
    print(f"sum={stats_arr.sum()}")
    print(f"mean={stats_arr.mean():.4f}")
    print(f"max={stats_arr.max()}, min={stats_arr.min()}")
    print(f"axis=0 sum={stats_arr.sum(axis=0)}")
    print(f"axis=1 mean={stats_arr.mean(axis=1)}")

    section("6) 和 Python list 的差异")
    py_list = [1, 2, 3]
    print(f"python list * 2 -> {py_list * 2}  (重复)")
    print(f"numpy array * 2 -> {(np.array(py_list) * 2)}  (逐元素乘法)")

    section("学习总结")
    print("1. ndarray 比 list 更适合数值计算，支持向量化。")
    print("2. shape / dtype / ndim 是理解数组的核心。")
    print("3. 常见操作：array / arange / linspace / reshape / astype / 聚合统计。")


if __name__ == "__main__":
    main()
