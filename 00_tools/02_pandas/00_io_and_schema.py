#!/usr/bin/env python3
"""
00_io_and_schema.py

pandas 第一课：IO 与 Schema 检查。

内容：
1. 构造示例数据并保存 CSV/Parquet
2. 读取数据并检查列类型
3. 常见类型修正（日期、数值、类别）
4. 输出基础数据质量报告

运行：
python3 00_io_and_schema.py
"""

from pathlib import Path
import pandas as pd
import numpy as np


def section(title: str):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def build_sample_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n = 12
    df = pd.DataFrame(
        {
            "user_id": [f"U{i:03d}" for i in range(1, n + 1)],
            "signup_date": pd.date_range("2026-01-01", periods=n, freq="7D").astype(str),
            "age": [22, 31, 27, 19, 45, 36, 28, 41, 33, 26, 29, 38],
            "city": ["beijing", "shanghai", "beijing", "guangzhou", "shenzhen", "beijing", "shanghai", "hangzhou", "nanjing", "beijing", "wuhan", "chengdu"],
            "income": [5200, 8900, 6400, 4300, 12000, 9700, 7300, 10600, 8800, 6900, 7600, 9900],
            # 故意做成字符串，模拟真实脏数据
            "is_paid": ["1", "0", "0", "1", "1", "0", "1", "1", "0", "0", "1", "1"],
        }
    )
    return df


def schema_report(df: pd.DataFrame, name: str):
    section(f"Schema Report - {name}")
    print("shape:", df.shape)
    print("dtypes:")
    print(df.dtypes)
    print("\nmissing count:")
    print(df.isna().sum())


def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "users.csv"
    parquet_path = out_dir / "users.parquet"

    section("1) 构造样例数据")
    df = build_sample_dataframe()
    print(df.head())

    section("2) 写入 CSV / Parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_ok = True
        print(f"CSV saved: {csv_path}")
        print(f"Parquet saved: {parquet_path}")
    except Exception as e:
        parquet_ok = False
        print(f"CSV saved: {csv_path}")
        print("Parquet 写入失败（可能未安装 pyarrow/fastparquet）:")
        print(e)

    section("3) 读取并查看原始 schema")
    df_csv = pd.read_csv(csv_path)
    schema_report(df_csv, "raw_from_csv")

    section("4) 类型修正")
    # 日期列
    df_csv["signup_date"] = pd.to_datetime(df_csv["signup_date"], errors="coerce")
    # 布尔/标记列
    df_csv["is_paid"] = pd.to_numeric(df_csv["is_paid"], errors="coerce").astype("Int64")
    # 类别列
    df_csv["city"] = df_csv["city"].astype("category")
    # 收入列强制为数值
    df_csv["income"] = pd.to_numeric(df_csv["income"], errors="coerce")

    schema_report(df_csv, "typed_from_csv")

    section("5) 基础数据质量检查")
    print("唯一 user_id 数:", df_csv["user_id"].nunique())
    print("age 范围:", (df_csv["age"].min(), df_csv["age"].max()))
    print("income 描述统计:")
    print(df_csv["income"].describe())

    section("6) 可选：读取 Parquet 对照")
    if parquet_ok:
        df_pq = pd.read_parquet(parquet_path)
        schema_report(df_pq, "from_parquet")
    else:
        print("跳过 Parquet 读取（当前环境缺少 parquet 引擎）")

    section("学习总结")
    print("1. 读取数据后第一件事是看 schema 和缺失情况。")
    print("2. 真实项目里应显式做类型修正，避免隐式类型导致的 bug。")
    print("3. CSV 适合通用交换，Parquet 更适合分析和性能。")


if __name__ == "__main__":
    main()
