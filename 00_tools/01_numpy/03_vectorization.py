#!/usr/bin/env python3
"""
03_vectorization.py

NumPy 第四课：向量化计算（Vectorization）与性能对比。

运行：
python3 03_vectorization.py
"""

import time
import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def timer(func, *args, repeat=3, **kwargs):
    """简单计时器：返回平均耗时（秒）和结果。"""
    elapsed = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)
    return sum(elapsed) / len(elapsed), result


def square_sum_loop(x):
    total = 0.0
    for v in x:
        total += v * v
    return total


def square_sum_vectorized(x):
    return np.sum(x * x)


def sigmoid_loop(x):
    out = np.empty_like(x)
    for i, v in enumerate(x):
        out[i] = 1.0 / (1.0 + np.exp(-v))
    return out


def sigmoid_vectorized(x):
    return 1.0 / (1.0 + np.exp(-x))


def pairwise_distance_loop(a, b):
    """计算两个同长度向量的欧氏距离（loop 版）"""
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return np.sqrt(s)


def pairwise_distance_vectorized(a, b):
    d = a - b
    return np.sqrt(np.sum(d * d))


def main():
    np.random.seed(42)

    section("1) 向量化是什么")
    print("向量化 = 把 Python 层循环，改为 NumPy 的底层批量计算。")
    print("好处：代码更短、通常更快、可读性更强。")

    section("2) 示例A：平方和")
    n = 1_000_000
    x = np.random.randn(n).astype(np.float64)

    t_loop, r_loop = timer(square_sum_loop, x)
    t_vec, r_vec = timer(square_sum_vectorized, x)

    print(f"loop result={r_loop:.6f}, time={t_loop*1000:.2f} ms")
    print(f"vec  result={r_vec:.6f}, time={t_vec*1000:.2f} ms")
    print(f"speedup ~ {t_loop / max(t_vec, 1e-12):.1f}x")

    section("3) 示例B：Sigmoid 激活")
    t_loop, y_loop = timer(sigmoid_loop, x)
    t_vec, y_vec = timer(sigmoid_vectorized, x)

    max_diff = np.max(np.abs(y_loop - y_vec))
    print(f"max(|loop-vec|)={max_diff:.12f}")
    print(f"loop time={t_loop*1000:.2f} ms")
    print(f"vec  time={t_vec*1000:.2f} ms")
    print(f"speedup ~ {t_loop / max(t_vec, 1e-12):.1f}x")

    section("4) 示例C：欧氏距离")
    a = np.random.randn(500_000)
    b = np.random.randn(500_000)

    t_loop, d_loop = timer(pairwise_distance_loop, a, b)
    t_vec, d_vec = timer(pairwise_distance_vectorized, a, b)

    print(f"distance loop={d_loop:.6f}, vec={d_vec:.6f}")
    print(f"absolute diff={abs(d_loop - d_vec):.12f}")
    print(f"loop time={t_loop*1000:.2f} ms")
    print(f"vec  time={t_vec*1000:.2f} ms")
    print(f"speedup ~ {t_loop / max(t_vec, 1e-12):.1f}x")

    section("5) 向量化注意点")
    print("1. 向量化通常更快，但可能占用更多内存（中间数组）。")
    print("2. 大数组下要注意 dtype，float32 更省内存。")
    print("3. 先保证结果正确，再追求性能。")

    section("学习总结")
    print("1. NumPy 的核心优势就是批量数组计算。")
    print("2. 能向量化的地方尽量向量化。")
    print("3. 用计时验证优化收益，不要凭感觉。")


if __name__ == "__main__":
    main()
