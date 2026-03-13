"""
07_fastapi / 04_model_loading.py
模型加载与缓存示例。
"""

from functools import lru_cache
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI(title="Model Loading")


class ToyModel:
    def __init__(self, bias: float = 0.1) -> None:
        self.bias = bias

    def predict_proba(self, x1: float, x2: float) -> float:
        z = self.bias + 0.45 * x1 + 0.55 * x2
        return 1 / (1 + pow(2.71828, -z))


@lru_cache(maxsize=1)
def load_model() -> ToyModel:
    # 模拟昂贵加载（磁盘/网络）
    return ToyModel(bias=0.2)


@app.get("/predict")
def predict(x1: float, x2: float) -> dict:
    model = load_model()
    p = model.predict_proba(x1, x2)
    return {"score": round(p, 4), "pred": int(p >= 0.5)}


def main() -> None:
    client = TestClient(app)
    r1 = client.get("/predict", params={"x1": 0.4, "x2": 0.6})
    r2 = client.get("/predict", params={"x1": 1.2, "x2": 0.1})

    print("first:", r1.status_code, r1.json())
    print("second:", r2.status_code, r2.json())
    print("model cache info:", load_model.cache_info())
    print("[Done] 04_model_loading.py completed successfully.")


if __name__ == "__main__":
    main()
