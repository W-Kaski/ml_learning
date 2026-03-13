# 01_numpy

目标：掌握数组计算、广播、向量化和线代基础。

建议脚本数：8

建议文件：
1. 00_array_basics.py: 创建、dtype、shape、reshape
2. 01_indexing_masking.py: 切片、布尔掩码、高级索引
3. 02_broadcasting.py: 广播规则与常见陷阱
4. 03_vectorization.py: 循环改向量化
5. 04_linear_algebra.py: dot、matmul、逆、特征值基础
6. 05_random_simulation.py: 随机分布与可复现实验
7. 06_performance_compare.py: numpy vs python 循环性能
8. 07_numpy_mini_project.py: 纯 numpy 线性回归

学习重点：
- 为什么向量化更快
- 数据类型对性能和精度的影响
- 与 torch tensor 的转换边界

验收标准：
- 你能把常见 for 循环改写为向量化
- 你能解释广播错误并修复
