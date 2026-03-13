"""
09_capstone / 03_service_fastapi.py
信用风险推理 API（FastAPI）。

启动示例：
python3 -m uvicorn 03_service_fastapi:app --host 0.0.0.0 --port 8020
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_xgb.joblib")
API_TOKEN = "capstone-token"


class PredictIn(BaseModel):
    age: int = Field(..., ge=18, le=75)
    income: float = Field(..., ge=15000, le=300000)
    loan_amount: float = Field(..., ge=1000, le=250000)
    credit_utilization: float = Field(..., ge=0.0, le=1.0)
    late_payments_12m: float = Field(..., ge=0, le=15)
    region: str = Field(..., pattern="^(North|South|East|West)$")
    channel: str = Field(..., pattern="^(Branch|Online|Partner)$")


class PredictOut(BaseModel):
    score: float
    label: str


app = FastAPI(title="Credit Risk Inference Service")


def _check_auth(authorization: str | None) -> None:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")


def _load_package() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run 01_train_model.py first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_exists": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn, authorization: str | None = Header(default=None)) -> PredictOut:
    _check_auth(authorization)
    package = _load_package()
    model = package["model"]
    feature_columns = package["feature_columns"]
    threshold = package["threshold"]

    row = {
        "age": payload.age,
        "income": payload.income,
        "loan_amount": payload.loan_amount,
        "credit_utilization": payload.credit_utilization,
        "late_payments_12m": payload.late_payments_12m,
        "region": payload.region,
        "channel": payload.channel,
        "debt_to_income": payload.loan_amount / (payload.income + 1e-6),
        "payment_stress": payload.credit_utilization * (1 + payload.late_payments_12m / 5),
        "is_online": int(payload.channel == "Online"),
    }
    X = pd.DataFrame([row])
    X = pd.get_dummies(X, columns=["region", "channel"], drop_first=False)
    X = X.reindex(columns=feature_columns, fill_value=0)

    score = float(model.predict_proba(X)[:, 1][0])
    label = "high_risk" if score >= threshold else "low_risk"
    return PredictOut(score=round(score, 4), label=label)


def _self_test() -> None:
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    h = client.get("/health")
    p = client.post(
        "/predict",
        headers=headers,
        json={
            "age": 39,
            "income": 90000,
            "loan_amount": 45000,
            "credit_utilization": 0.42,
            "late_payments_12m": 1,
            "region": "East",
            "channel": "Online",
        },
    )

    print("/health:", h.status_code, h.json())
    print("/predict:", p.status_code, p.json())


if __name__ == "__main__":
    _self_test()
    print("[Done] 03_service_fastapi.py completed successfully.")
