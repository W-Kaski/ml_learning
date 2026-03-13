#!/usr/bin/env python3
"""
04_linear_algebra.py

NumPy 第五课：线性代数基础（工作常用版）。

运行：
python3 04_linear_algebra.py
"""

import numpy as np


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(42)

    section("1) 向量与矩阵的点积 / 矩阵乘法")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print("a:", a)
    print("b:", b)
    print("dot(a, b) =", np.dot(a, b))

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    print("\nA:\n", A)
    print("B:\n", B)
    print("A @ B:\n", A @ B)

    section("2) 转置 transpose")
    M = np.array([[1, 2, 3], [4, 5, 6]])
    print("M:\n", M)
    print("M.T:\n", M.T)

    section("3) 行列式 det 与可逆性")
    C = np.array([[2.0, 1.0], [5.0, 3.0]])
    det_c = np.linalg.det(C)
    print("C:\n", C)
    print(f"det(C) = {det_c:.6f}")
    print("det != 0 => C 可逆")

    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    det_s = np.linalg.det(singular)
    print("\nsingular:\n", singular)
    print(f"det(singular) = {det_s:.6f}")
    print("det = 0 => 不可逆")

    section("4) 逆矩阵 inverse 与线性方程组")
    d = np.array([1.0, 2.0])
    C_inv = np.linalg.inv(C)
    x_via_inv = C_inv @ d
    x_via_solve = np.linalg.solve(C, d)

    print("C_inv:\n", C_inv)
    print("x = inv(C) @ d ->", x_via_inv)
    print("x = solve(C, d) ->", x_via_solve)
    print("验证 C @ x =", C @ x_via_solve)
    print("提示：数值稳定性上，通常优先用 solve 而非显式求逆")

    section("5) 秩 rank")
    R1 = np.array([[1.0, 2.0], [2.0, 4.0]])
    R2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    print("R1:\n", R1)
    print("rank(R1) =", np.linalg.matrix_rank(R1))
    print("R2:\n", R2)
    print("rank(R2) =", np.linalg.matrix_rank(R2))

    section("6) 特征值 / 特征向量 eig")
    E = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigvals, eigvecs = np.linalg.eig(E)
    print("E:\n", E)
    print("eigvals:", eigvals)
    print("eigvecs:\n", eigvecs)

    # 验证 A v = lambda v（取第一个特征对）
    lam = eigvals[0]
    v = eigvecs[:, 0]
    left = E @ v
    right = lam * v
    print("\n验证 A v 与 lambda v 差值:", np.linalg.norm(left - right))

    section("7) 奇异值分解 SVD")
    X = np.array(
        [
            [3.0, 1.0, 1.0],
            [-1.0, 3.0, 1.0],
        ]
    )
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    print("X:\n", X)
    print("U:\n", U)
    print("S:", S)
    print("VT:\n", VT)

    X_recon = U @ np.diag(S) @ VT
    print("\n重构误差 ||X - U S VT|| =", np.linalg.norm(X - X_recon))

    section("8) 最小二乘（线性回归基础）")
    # y ≈ w0 + w1*x
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([1.1, 3.0, 5.1, 7.2, 9.1])

    # 构建设计矩阵 [1, x]
    X_design = np.column_stack([np.ones_like(x_data), x_data])
    w, *_ = np.linalg.lstsq(X_design, y_data, rcond=None)
    print("X_design:\n", X_design)
    print("y:", y_data)
    print("最小二乘解 w=[w0,w1] ->", w)

    y_pred = X_design @ w
    mse = np.mean((y_pred - y_data) ** 2)
    print("y_pred:", y_pred)
    print(f"MSE={mse:.6f}")

    section("学习总结")
    print("1. @ 是矩阵乘法，dot 对一维向量是内积。")
    print("2. 线性方程组建议用 solve，不要优先手动求逆。")
    print("3. SVD / 特征分解是很多 ML 算法的基础。")
    print("4. lstsq 是理解线性回归参数估计的关键工具。")


if __name__ == "__main__":
    main()
