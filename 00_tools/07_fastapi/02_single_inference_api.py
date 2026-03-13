"""
07_fastapi / 02_single_inference_api.py
单条推理接口示例。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI(title="Single Inference API")


class InferenceIn(BaseModel):
    feature_1: float = Field(..., ge=-10, le=10)
    feature_2: float = Field(..., ge=-10, le=10)


class InferenceOut(BaseModel):
    score: float
    pred: int


@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn) -> InferenceOut:
    # toy linear model
    score = 0.35 * payload.feature_1 + 0.65 * payload.feature_2
    prob = 1 / (1 + pow(2.71828, -score))
    return InferenceOut(score=round(prob, 4), pred=int(prob >= 0.5))


def main() -> None:
    client = TestClient(app)
    resp = client.post("/infer", json={"feature_1": 1.2, "feature_2": 0.8})
    print("/infer:", resp.status_code, resp.json())
    print("[Done] 02_single_inference_api.py completed successfully.")


if __name__ == "__main__":
    main()
