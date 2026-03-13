"""
07_fastapi / 00_fastapi_basics.py
路由和入参校验基础。
"""

from fastapi import FastAPI, Query
from fastapi.testclient import TestClient

app = FastAPI(title="FastAPI Basics")


@app.get("/")
def root() -> dict:
    return {"message": "hello fastapi"}


@app.get("/add")
def add(a: int = Query(..., ge=0), b: int = Query(..., ge=0)) -> dict:
    return {"a": a, "b": b, "sum": a + b}


def main() -> None:
    client = TestClient(app)
    r1 = client.get("/")
    r2 = client.get("/add", params={"a": 3, "b": 7})
    r3 = client.get("/add", params={"a": -1, "b": 2})

    print("/ status:", r1.status_code, r1.json())
    print("/add valid:", r2.status_code, r2.json())
    print("/add invalid:", r3.status_code)
    print("[Done] 00_fastapi_basics.py completed successfully.")


if __name__ == "__main__":
    main()
