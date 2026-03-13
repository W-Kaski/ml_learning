"""
05_xgboost / 06_xgb_project_report.py
小项目：与 sklearn 随机森林对比，并输出调参与收益结论。
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    rf_pred = (rf_proba >= 0.5).astype(int)
    xgb_pred = (xgb_proba >= 0.5).astype(int)

    rf_auc = roc_auc_score(y_test, rf_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    print("[Project Comparison]")
    print(f"RF  CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    print(f"XGB CV AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
    print(f"RF  Test AUC: {rf_auc:.4f}  ACC: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"XGB Test AUC: {xgb_auc:.4f}  ACC: {accuracy_score(y_test, xgb_pred):.4f}")

    gain_auc = xgb_auc - rf_auc
    print("\n[Conclusion]")
    if gain_auc >= 0:
        print(f"XGBoost 相比 RandomForest 测试集 AUC 提升: +{gain_auc:.4f}")
    else:
        print(f"XGBoost 相比 RandomForest 测试集 AUC 下降: {gain_auc:.4f}")

    print("推荐调参路径:")
    print("1) 先定 learning_rate + n_estimators")
    print("2) 再调 max_depth + min_child_weight + gamma")
    print("3) 最后调 subsample + colsample_bytree + reg_lambda")
    print("[Done] 06_xgb_project_report.py completed successfully.")


if __name__ == "__main__":
    main()
