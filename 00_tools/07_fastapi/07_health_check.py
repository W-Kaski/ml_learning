"""
07_fastapi / 07_health_check.py
健康检查接口。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI(title="Health Check")


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict:
    # 可扩展数据库、模型、缓存等依赖检查
    checks = {"model_loaded": True, "db_connected": True}
    ok = all(checks.values())
    return {"status": "ready" if ok else "not_ready", "checks": checks}


def main() -> None:
    client = TestClient(app)
    r1 = client.get("/health/live")
    r2 = client.get("/health/ready")
    print("live:", r1.status_code, r1.json())
    print("ready:", r2.status_code, r2.json())
    print("[Done] 07_health_check.py completed successfully.")


if __name__ == "__main__":
    main()
