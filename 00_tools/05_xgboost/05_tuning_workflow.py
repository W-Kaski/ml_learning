"""
05_xgboost / 05_tuning_workflow.py
调参工作流：
1) 固定树数观察学习率
2) 再调深度和叶子复杂度
3) 最后微调采样参数
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Step 1: 粗搜 learning_rate + n_estimators
    search1 = RandomizedSearchCV(
        estimator=base,
        param_distributions={
            "n_estimators": [200, 300, 500, 800],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        },
        n_iter=10,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    search1.fit(X_train, y_train)

    # Step 2: 基于 step1 最优，再搜结构参数
    best1 = search1.best_params_
    model2 = XGBClassifier(
        **best1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    search2 = RandomizedSearchCV(
        estimator=model2,
        param_distributions={
            "max_depth": [2, 3, 4, 5, 6, 8],
            "min_child_weight": [1, 2, 4, 6],
            "gamma": [0, 0.1, 0.3, 0.5, 1.0],
        },
        n_iter=12,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    search2.fit(X_train, y_train)

    # Step 3: 采样参数
    best2 = {**best1, **search2.best_params_}
    model3 = XGBClassifier(
        **best2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    search3 = RandomizedSearchCV(
        estimator=model3,
        param_distributions={
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.1, 1, 3, 5, 10],
        },
        n_iter=12,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    search3.fit(X_train, y_train)

    best = {**best2, **search3.best_params_}
    final_model = XGBClassifier(
        **best,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    proba = final_model.predict_proba(X_test)[:, 1]
    pred = final_model.predict(X_test)

    print("Step1 best:", best1, "cv_auc=", round(search1.best_score_, 4))
    print("Step2 best:", search2.best_params_, "cv_auc=", round(search2.best_score_, 4))
    print("Step3 best:", search3.best_params_, "cv_auc=", round(search3.best_score_, 4))
    print("\nFinal test AUC:", round(roc_auc_score(y_test, proba), 4))
    print("Final test ACC:", round(accuracy_score(y_test, pred), 4))
    print("[Done] 05_tuning_workflow.py completed successfully.")


if __name__ == "__main__":
    main()
