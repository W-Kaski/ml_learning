"""
07_fastapi / 01_pydantic_schema.py
请求/响应模型定义。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI(title="Pydantic Schema")


class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    income: float = Field(..., ge=0)
    vip: bool = False


class PredictResponse(BaseModel):
    score: float
    label: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    score = min(0.99, 0.1 + req.income / 200000 + (0.2 if req.vip else 0.0) + req.age / 500)
    label = "buy" if score >= 0.5 else "not_buy"
    return PredictResponse(score=round(score, 4), label=label)


def main() -> None:
    client = TestClient(app)

    ok = client.post("/predict", json={"age": 30, "income": 80000, "vip": True})
    bad = client.post("/predict", json={"age": -3, "income": 80000, "vip": True})

    print("valid:", ok.status_code, ok.json())
    print("invalid:", bad.status_code)
    print("[Done] 01_pydantic_schema.py completed successfully.")


if __name__ == "__main__":
    main()
