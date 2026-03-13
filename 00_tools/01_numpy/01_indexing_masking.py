#!/usr/bin/env python3
"""
01_indexing_masking.py

NumPy 第二课：索引、切片、布尔掩码、高级索引。

运行：
python3 01_indexing_masking.py
"""

import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    np.random.seed(42)

    section("1) 一维数组索引与切片")
    arr1d = np.array([10, 20, 30, 40, 50, 60])
    print("arr1d:", arr1d)
    print("arr1d[0] =", arr1d[0])
    print("arr1d[-1] =", arr1d[-1])
    print("arr1d[1:4] =", arr1d[1:4])
    print("arr1d[:3] =", arr1d[:3])
    print("arr1d[::2] =", arr1d[::2])

    section("2) 二维数组索引")
    arr2d = np.arange(1, 13).reshape(3, 4)
    print("arr2d:\n", arr2d)
    print("arr2d[0, 0] =", arr2d[0, 0])
    print("arr2d[2, 3] =", arr2d[2, 3])
    print("arr2d[1] =", arr2d[1])
    print("arr2d[:, 2] =", arr2d[:, 2])
    print("arr2d[0:2, 1:3] =\n", arr2d[0:2, 1:3])

    section("3) 布尔掩码（Boolean Masking）")
    scores = np.array([58, 72, 91, 66, 85, 49, 77])
    print("scores:", scores)

    passed_mask = scores >= 60
    print("passed_mask:", passed_mask)
    print("scores[passed_mask] =", scores[passed_mask])

    # 常见写法：直接把条件放到 [] 里
    print("scores[scores >= 80] =", scores[scores >= 80])

    section("4) 多条件筛选")
    values = np.array([5, 12, 18, 23, 31, 42, 55])
    print("values:", values)
    cond = (values >= 15) & (values <= 40)
    print("15 <= values <= 40:", values[cond])

    # 注意：用 & 和 |，不是 and / or
    odd_or_big = (values % 2 == 1) | (values > 40)
    print("odd or > 40:", values[odd_or_big])

    section("5) 高级索引（Fancy Indexing）")
    x = np.array([100, 200, 300, 400, 500])
    idx = np.array([0, 2, 4])
    print("x:", x)
    print("idx:", idx)
    print("x[idx] =", x[idx])

    y = np.arange(1, 10).reshape(3, 3)
    row_idx = np.array([0, 1, 2])
    col_idx = np.array([2, 1, 0])
    print("y:\n", y)
    print("y[row_idx, col_idx] =", y[row_idx, col_idx])

    section("6) 原地修改与拷贝注意点")
    base = np.array([1, 2, 3, 4, 5])
    sliced_view = base[1:4]   # 视图 view
    sliced_view[0] = 999
    print("base after sliced_view modify:", base)

    copied = base[[0, 2, 4]]  # 高级索引返回拷贝 copy
    copied[0] = -1
    print("copied:", copied)
    print("base after copied modify:", base)

    section("学习总结")
    print("1. 切片通常返回 view，高级索引通常返回 copy。")
    print("2. 布尔掩码是数据筛选最常用方式。")
    print("3. 多条件筛选要用 & 和 |，并加括号。")


if __name__ == "__main__":
    main()
