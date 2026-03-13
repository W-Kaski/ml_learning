"""
07_fastapi / 08_simple_auth.py
简单鉴权（Bearer Token）。
"""

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

app = FastAPI(title="Simple Auth")
API_TOKEN = "ml-learning-token"


def verify_token(authorization: str | None) -> None:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or invalid auth header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")


@app.get("/secure/predict")
def secure_predict(x: float, authorization: str | None = Header(default=None)) -> dict:
    verify_token(authorization)
    return {"x": x, "y": round(0.8 * x + 0.2, 4)}


def main() -> None:
    client = TestClient(app)
    no_auth = client.get("/secure/predict", params={"x": 1.5})
    bad_auth = client.get(
        "/secure/predict",
        params={"x": 1.5},
        headers={"Authorization": "Bearer wrong-token"},
    )
    ok_auth = client.get(
        "/secure/predict",
        params={"x": 1.5},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )

    print("no auth:", no_auth.status_code, no_auth.json())
    print("bad auth:", bad_auth.status_code, bad_auth.json())
    print("ok auth:", ok_auth.status_code, ok_auth.json())
    print("[Done] 08_simple_auth.py completed successfully.")


if __name__ == "__main__":
    main()
