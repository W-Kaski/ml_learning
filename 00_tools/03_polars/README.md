# 03_polars

目标：掌握高性能数据处理，理解 lazy 执行与表达式 API。

建议脚本数：8

建议文件：
1. 00_polars_basics.py: DataFrame/LazyFrame 入门
2. 01_expressions.py: 表达式列运算
3. 02_groupby_window.py: 分组聚合与窗口函数
4. 03_joins.py: 各类 join 与性能注意点
5. 04_lazy_pipeline.py: scan + lazy 查询链
6. 05_parquet_workflow.py: parquet 读写与列裁剪
7. 06_polars_vs_pandas.py: 同任务性能对比
8. 07_polars_mini_project.py: 完整数据管道

学习重点：
- eager 和 lazy 的差异
- 查询计划优化思路
- 何时选 pandas，何时选 polars

验收标准：
- 能把一段 pandas 管道迁移为 polars
- 能解释性能提升来自哪里
