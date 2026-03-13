#!/usr/bin/env python3
"""
07_numpy_mini_project.py

NumPy 收官项目：纯 NumPy 线性回归（不依赖 sklearn / torch）。

内容：
1. 合成数据生成
2. 特征标准化
3. 梯度下降训练线性回归
4. 训练过程指标记录
5. 测试集评估

运行：
python3 07_numpy_mini_project.py
"""

import json
from pathlib import Path

import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def train_test_split(x, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    test_size = int(len(x) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_fit(x_train):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def standardize_transform(x, mean, std):
    return (x - mean) / std


def add_bias(x):
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def gradient_descent_linear_regression(x, y, lr=0.05, epochs=1000):
    """
    最小化 MSE:
      J(w) = (1/n) * ||Xw - y||^2
    梯度:
      dJ/dw = (2/n) * X^T (Xw - y)
    """
    n, d = x.shape
    w = np.zeros((d, 1))
    y = y.reshape(-1, 1)

    history = []
    for epoch in range(1, epochs + 1):
        pred = x @ w
        err = pred - y
        grad = (2.0 / n) * (x.T @ err)
        w -= lr * grad

        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            loss = np.mean(err ** 2)
            history.append({"epoch": epoch, "train_mse": float(loss)})

    return w, history


def main():
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(42)

    section("1) 生成合成数据")
    n_samples = 2000
    n_features = 3

    x = np.random.randn(n_samples, n_features)
    true_w = np.array([[3.5], [-2.0], [1.2]])
    bias = 4.0
    noise = np.random.normal(0, 0.5, size=(n_samples, 1))
    y = x @ true_w + bias + noise

    print(f"x.shape={x.shape}, y.shape={y.shape}")
    print(f"真实参数: bias={bias}, w={true_w.ravel()}")

    section("2) 切分训练/测试 + 标准化")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.2, seed=42)
    mean, std = standardize_fit(x_train)
    x_train_std = standardize_transform(x_train, mean, std)
    x_test_std = standardize_transform(x_test, mean, std)

    x_train_bias = add_bias(x_train_std)
    x_test_bias = add_bias(x_test_std)

    print(f"train={x_train_bias.shape}, test={x_test_bias.shape}")

    section("3) 训练（梯度下降）")
    learned_w, history = gradient_descent_linear_regression(
        x_train_bias,
        y_train,
        lr=0.05,
        epochs=1200,
    )
    print("训练完成。")
    print("学到参数（标准化空间）:", learned_w.ravel())
    print("训练过程（抽样点）:")
    for item in history[:3]:
        print(item)
    print("...")
    for item in history[-3:]:
        print(item)

    section("4) 评估")
    y_train_pred = x_train_bias @ learned_w
    y_test_pred = x_test_bias @ learned_w

    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)
    train_mae = mae(y_train, y_train_pred)
    test_mae = mae(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train: MSE={train_mse:.4f}, MAE={train_mae:.4f}, R2={train_r2:.4f}")
    print(f"Test : MSE={test_mse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}")

    section("5) 保存结果")
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "n_samples": n_samples,
        "n_features": n_features,
        "true_bias": bias,
        "true_w": true_w.ravel().tolist(),
        "learned_w_standardized": learned_w.ravel().tolist(),
        "train_metrics": {
            "mse": float(train_mse),
            "mae": float(train_mae),
            "r2": float(train_r2),
        },
        "test_metrics": {
            "mse": float(test_mse),
            "mae": float(test_mae),
            "r2": float(test_r2),
        },
        "history": history,
    }

    out_path = out_dir / "linear_regression_result.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_path}")

    section("学习总结")
    print("1. 仅用 NumPy 就能完整实现一个监督学习训练流程。")
    print("2. 标准化 + 梯度下降是很多模型训练的核心模式。")
    print("3. MSE/MAE/R2 三个指标可以从不同角度看拟合质量。")


if __name__ == "__main__":
    main()
