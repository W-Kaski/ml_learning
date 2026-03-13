#!/usr/bin/env python3
"""
06_performance_compare.py

NumPy 第七课：性能基准（loop/list comprehension/vectorized）与内存对比。

运行：
python3 06_performance_compare.py
"""

import time
import sys
import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def bench(func, *args, repeat=3, **kwargs):
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times), out


def py_loop_square_sum(data):
    s = 0.0
    for v in data:
        s += v * v
    return s


def py_listcomp_square_sum(data):
    return sum([v * v for v in data])


def np_vectorized_square_sum(data_np):
    return np.sum(data_np * data_np)


def py_loop_affine(data, a=1.7, b=0.3):
    out = []
    for v in data:
        out.append(a * v + b)
    return out


def py_listcomp_affine(data, a=1.7, b=0.3):
    return [a * v + b for v in data]


def np_vectorized_affine(data_np, a=1.7, b=0.3):
    return a * data_np + b


def bytes_to_mb(n_bytes):
    return n_bytes / (1024 ** 2)


def main():
    np.random.seed(42)

    section("1) 构造测试数据")
    n = 1_000_000
    data_np = np.random.randn(n).astype(np.float64)
    data_list = data_np.tolist()
    print(f"样本数 n={n:,}")
    print(f"numpy dtype={data_np.dtype}, shape={data_np.shape}")

    section("2) 任务A：平方和 sum(x^2)")
    t_loop, r_loop = bench(py_loop_square_sum, data_list)
    t_listcomp, r_listcomp = bench(py_listcomp_square_sum, data_list)
    t_np, r_np = bench(np_vectorized_square_sum, data_np)

    print(f"python loop      : {t_loop*1000:.2f} ms, result={r_loop:.6f}")
    print(f"list comprehension: {t_listcomp*1000:.2f} ms, result={r_listcomp:.6f}")
    print(f"numpy vectorized : {t_np*1000:.2f} ms, result={r_np:.6f}")
    print(f"loop vs numpy speedup      ~ {t_loop / max(t_np, 1e-12):.1f}x")
    print(f"listcomp vs numpy speedup  ~ {t_listcomp / max(t_np, 1e-12):.1f}x")

    section("3) 任务B：仿射变换 y = a*x + b")
    t_loop, out_loop = bench(py_loop_affine, data_list)
    t_listcomp, out_listcomp = bench(py_listcomp_affine, data_list)
    t_np, out_np = bench(np_vectorized_affine, data_np)

    # 只取前5项验证
    print(f"python loop      : {t_loop*1000:.2f} ms, head={out_loop[:5]}")
    print(f"list comprehension: {t_listcomp*1000:.2f} ms, head={out_listcomp[:5]}")
    print(f"numpy vectorized : {t_np*1000:.2f} ms, head={out_np[:5]}")

    max_diff = np.max(np.abs(np.array(out_loop[:1000]) - out_np[:1000]))
    print(f"前1000项 loop 与 numpy 最大差值: {max_diff:.12f}")

    section("4) dtype 与内存")
    x64 = np.random.randn(2_000_000).astype(np.float64)
    x32 = x64.astype(np.float32)

    print(f"float64 nbytes={x64.nbytes:,} ({bytes_to_mb(x64.nbytes):.2f} MB)")
    print(f"float32 nbytes={x32.nbytes:,} ({bytes_to_mb(x32.nbytes):.2f} MB)")
    print("float32 大约节省 50% 内存")

    t64, r64 = bench(np_vectorized_square_sum, x64)
    t32, r32 = bench(np_vectorized_square_sum, x32)
    print(f"float64 sum(x^2) time={t64*1000:.2f} ms")
    print(f"float32 sum(x^2) time={t32*1000:.2f} ms")
    print(f"结果差异 |r64-r32|={abs(float(r64)-float(r32)):.4f}")

    section("5) 性能实验注意事项")
    print("1. 先做正确性验证，再比较性能。")
    print("2. 多次 repeat 取平均，避免单次抖动。")
    print("3. 对大数组，dtype 与内存带宽同样影响性能。")
    print("4. 极端优化前先定位真正瓶颈。")

    section("学习总结")
    print("1. NumPy 向量化通常显著快于 Python 循环。")
    print("2. 列表推导比 for-loop 好，但仍常慢于 NumPy。")
    print("3. 选择 float32 / float64 是性能与精度权衡。")


if __name__ == "__main__":
    main()
