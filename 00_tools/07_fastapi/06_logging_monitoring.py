"""
07_fastapi / 06_logging_monitoring.py
请求日志中间件。
"""

import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

app = FastAPI(title="Logging Monitoring")


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    print(
        f"[REQ] method={request.method} path={request.url.path} "
        f"status={response.status_code} latency_ms={latency_ms:.2f}"
    )
    response.headers["X-Latency-MS"] = f"{latency_ms:.2f}"
    return response


@app.get("/ping")
def ping() -> JSONResponse:
    return JSONResponse(content={"message": "pong"})


def main() -> None:
    client = TestClient(app)
    r = client.get("/ping")
    print("status:", r.status_code, "header latency:", r.headers.get("X-Latency-MS"))
    print("[Done] 06_logging_monitoring.py completed successfully.")


if __name__ == "__main__":
    main()
