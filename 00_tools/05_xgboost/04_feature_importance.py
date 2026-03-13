"""
05_xgboost / 04_feature_importance.py
特征重要性分析（gain/weight）并对比 permutation importance。
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    weight_dict = booster.get_score(importance_type="weight")

    gain_series = pd.Series(gain_dict, name="gain").sort_values(ascending=False)
    weight_series = pd.Series(weight_dict, name="weight").sort_values(ascending=False)

    print("Top-10 by gain:")
    print(gain_series.head(10))

    print("\nTop-10 by weight:")
    print(weight_series.head(10))

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "perm_importance": perm.importances_mean,
    }).sort_values("perm_importance", ascending=False)

    print("\nTop-10 by permutation importance (AUC drop):")
    print(perm_df.head(10).to_string(index=False))
    print("[Done] 04_feature_importance.py completed successfully.")


if __name__ == "__main__":
    main()
