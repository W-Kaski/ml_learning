#!/usr/bin/env python3
"""
05_random_simulation.py

NumPy 第六课：随机数、常见分布与蒙特卡洛模拟。

运行：
python3 05_random_simulation.py
"""

import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def summary_stats(name: str, x: np.ndarray):
    print(
        f"{name}: mean={x.mean():.4f}, std={x.std():.4f}, "
        f"min={x.min():.4f}, max={x.max():.4f}"
    )


def main():
    section("1) 随机种子与可复现")
    np.random.seed(42)
    a1 = np.random.rand(5)
    np.random.seed(42)
    a2 = np.random.rand(5)
    print("a1:", a1)
    print("a2:", a2)
    print("a1 与 a2 完全相同 ->", np.allclose(a1, a2))

    section("2) 常见分布采样")
    np.random.seed(42)
    normal = np.random.normal(loc=0.0, scale=1.0, size=100_000)
    uniform = np.random.uniform(low=-1.0, high=1.0, size=100_000)
    binomial = np.random.binomial(n=10, p=0.3, size=100_000)

    summary_stats("normal(0,1)", normal)
    summary_stats("uniform(-1,1)", uniform)
    summary_stats("binomial(10,0.3)", binomial)

    section("3) 抽样实验：大数定律直觉")
    np.random.seed(42)
    coin = np.random.binomial(n=1, p=0.5, size=10_000)
    running_mean = np.cumsum(coin) / np.arange(1, len(coin) + 1)
    print("前10次均值:", np.round(running_mean[:10], 4))
    print("第100次均值:", round(running_mean[99], 4))
    print("第1000次均值:", round(running_mean[999], 4))
    print("第10000次均值:", round(running_mean[-1], 4))

    section("4) 蒙特卡洛模拟：估计 pi")
    np.random.seed(42)
    n = 500_000
    x = np.random.uniform(-1.0, 1.0, size=n)
    y = np.random.uniform(-1.0, 1.0, size=n)
    inside_circle = (x**2 + y**2) <= 1.0
    pi_est = 4.0 * inside_circle.mean()
    print(f"sample_size={n}")
    print(f"pi_estimate={pi_est:.6f}")
    print(f"abs_error={abs(np.pi - pi_est):.6f}")

    section("5) 置信区间（简单近似）")
    # 对伯努利分布样本均值，近似 95% CI: p_hat ± 1.96 * sqrt(p_hat*(1-p_hat)/n)
    np.random.seed(42)
    n_ci = 20_000
    p_true = 0.3
    samples = np.random.binomial(n=1, p=p_true, size=n_ci)
    p_hat = samples.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / n_ci)
    ci_low = p_hat - 1.96 * se
    ci_high = p_hat + 1.96 * se
    print(f"p_true={p_true}, p_hat={p_hat:.4f}")
    print(f"95% CI ~ [{ci_low:.4f}, {ci_high:.4f}]")

    section("学习总结")
    print("1. 随机种子是复现实验的基础。")
    print("2. 分布采样是模拟和建模的核心工具。")
    print("3. 蒙特卡洛方法可用于难解析问题的数值估计。")
    print("4. 样本数增大时，统计量会更稳定。")


if __name__ == "__main__":
    main()
