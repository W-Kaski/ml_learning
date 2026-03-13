"""
09_capstone / 01_train_model.py
训练信用风险模型并导出模型与预测结果。
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "credit_risk_processed.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "credit_risk_xgb.joblib")
METRICS_PATH = os.path.join(OUT_DIR, "train_metrics.json")
PRED_PATH = os.path.join(OUT_DIR, "test_predictions.csv")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run 00_data_pipeline.py first.")

    df = pd.read_csv(DATA_PATH)

    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    X = pd.get_dummies(X, columns=["region", "channel"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }

    package = {
        "model": model,
        "feature_columns": list(X.columns),
        "threshold": 0.5,
    }
    joblib.dump(package, MODEL_PATH)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.to_numpy()
    pred_df["y_pred"] = pred
    pred_df["y_proba"] = proba
    pred_df.to_csv(PRED_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("model saved:", MODEL_PATH)
    print("metrics saved:", METRICS_PATH)
    print("auc:", round(metrics["auc"], 4), "f1:", round(metrics["f1"], 4))
    print("[Done] 01_train_model.py completed successfully.")


if __name__ == "__main__":
    main()
