"""
07_fastapi / 03_batch_inference_api.py
批量推理接口示例。
"""

from typing import List
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI(title="Batch Inference API")


class Item(BaseModel):
    x1: float = Field(..., ge=-10, le=10)
    x2: float = Field(..., ge=-10, le=10)


class BatchIn(BaseModel):
    items: List[Item] = Field(..., min_length=1, max_length=1000)


class ItemOut(BaseModel):
    score: float
    pred: int


class BatchOut(BaseModel):
    n: int
    results: List[ItemOut]


@app.post("/batch_infer", response_model=BatchOut)
def batch_infer(payload: BatchIn) -> BatchOut:
    outs: List[ItemOut] = []
    for it in payload.items:
        score = 0.4 * it.x1 + 0.6 * it.x2
        prob = 1 / (1 + pow(2.71828, -score))
        outs.append(ItemOut(score=round(prob, 4), pred=int(prob >= 0.5)))
    return BatchOut(n=len(outs), results=outs)


def main() -> None:
    client = TestClient(app)
    resp = client.post(
        "/batch_infer",
        json={"items": [{"x1": 0.1, "x2": 0.2}, {"x1": 1.0, "x2": 2.0}, {"x1": -0.5, "x2": 0.3}]},
    )
    print("status:", resp.status_code)
    print("result:", resp.json())
    print("[Done] 03_batch_inference_api.py completed successfully.")


if __name__ == "__main__":
    main()
