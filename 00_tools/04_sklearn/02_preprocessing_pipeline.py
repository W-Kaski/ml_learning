"""
04_sklearn / 02_preprocessing_pipeline.py
使用 ColumnTransformer + Pipeline 处理数值/类别特征。
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def make_dataset(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 60, n)
    income = rng.normal(80000, 18000, n)
    city = rng.choice(["BJ", "SH", "GZ", "SZ"], n)
    vip = rng.choice(["Y", "N"], n, p=[0.3, 0.7])

    # 构造标签：收入高 + vip + 年龄区间更可能购买
    logit = -6 + 0.00006 * income + 0.8 * (vip == "Y") + 0.03 * (age - 30)
    p = 1 / (1 + np.exp(-logit))
    bought = rng.binomial(1, p)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "city": city,
        "vip": vip,
        "bought": bought,
    })

    # 注入少量缺失
    miss_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    df.loc[miss_idx, "income"] = np.nan
    return df


def main() -> None:
    df = make_dataset()
    X = df.drop(columns=["bought"])
    y = df["bought"]

    num_cols = ["age", "income"]
    cat_cols = ["city", "vip"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    clf = Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("rows:", len(df), "positive rate:", round(y.mean(), 4))
    print("pipeline accuracy:", round(accuracy_score(y_test, pred), 4))
    print("[Done] 02_preprocessing_pipeline.py completed successfully.")


if __name__ == "__main__":
    main()
