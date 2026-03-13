"""
07_fastapi / 09_api_project.py
可运行 FastAPI 服务项目（含本地测试）。

启动命令：
python3 -m uvicorn 09_api_project:app --host 0.0.0.0 --port 8000 --reload
"""

from functools import lru_cache
from typing import List
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

app = FastAPI(title="ML Inference Service", version="1.0.0")
API_TOKEN = "ml-learning-token"


class OnePredictIn(BaseModel):
    age: int = Field(..., ge=0, le=120)
    income: float = Field(..., ge=0)
    vip: bool = False


class OnePredictOut(BaseModel):
    score: float
    label: str


class BatchPredictIn(BaseModel):
    items: List[OnePredictIn] = Field(..., min_length=1, max_length=200)


class BatchPredictOut(BaseModel):
    n: int
    results: List[OnePredictOut]


class ToyModel:
    def predict_score(self, age: int, income: float, vip: bool) -> float:
        raw = -3.5 + income / 50000 + age / 80 + (0.6 if vip else 0)
        prob = 1 / (1 + pow(2.71828, -raw))
        return float(max(0.0, min(1.0, prob)))


@lru_cache(maxsize=1)
def get_model() -> ToyModel:
    return ToyModel()


def check_auth(authorization: str | None) -> None:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")


@app.get("/")
def root() -> dict:
    return {"service": "ml-inference", "version": app.version}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_cached": get_model.cache_info().currsize == 1}


@app.post("/predict", response_model=OnePredictOut)
def predict(payload: OnePredictIn, authorization: str | None = Header(default=None)) -> OnePredictOut:
    check_auth(authorization)
    model = get_model()
    score = model.predict_score(payload.age, payload.income, payload.vip)
    label = "buy" if score >= 0.5 else "not_buy"
    return OnePredictOut(score=round(score, 4), label=label)


@app.post("/predict_batch", response_model=BatchPredictOut)
def predict_batch(payload: BatchPredictIn, authorization: str | None = Header(default=None)) -> BatchPredictOut:
    check_auth(authorization)
    model = get_model()

    out: List[OnePredictOut] = []
    for item in payload.items:
        score = model.predict_score(item.age, item.income, item.vip)
        out.append(OnePredictOut(score=round(score, 4), label="buy" if score >= 0.5 else "not_buy"))

    return BatchPredictOut(n=len(out), results=out)


def _self_test() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    token_header = {"Authorization": f"Bearer {API_TOKEN}"}

    r1 = client.get("/")
    r2 = client.get("/health")
    r3 = client.post("/predict", json={"age": 35, "income": 90000, "vip": True}, headers=token_header)
    r4 = client.post(
        "/predict_batch",
        json={
            "items": [
                {"age": 25, "income": 60000, "vip": False},
                {"age": 45, "income": 120000, "vip": True},
            ]
        },
        headers=token_header,
    )

    print("/:", r1.status_code, r1.json())
    print("/health:", r2.status_code, r2.json())
    print("/predict:", r3.status_code, r3.json())
    print("/predict_batch:", r4.status_code, r4.json())


if __name__ == "__main__":
    _self_test()
    print("[Done] 09_api_project.py completed successfully.")
