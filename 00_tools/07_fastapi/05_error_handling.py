"""
07_fastapi / 05_error_handling.py
异常处理和返回码。
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

app = FastAPI(title="Error Handling")


class ModelNotReadyError(Exception):
    pass


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(_: Request, exc: ModelNotReadyError):
    return JSONResponse(status_code=503, content={"error": str(exc), "code": "MODEL_NOT_READY"})


@app.get("/item/{item_id}")
def get_item(item_id: int) -> dict:
    if item_id < 0:
        raise HTTPException(status_code=400, detail="item_id must be non-negative")
    if item_id == 999:
        raise ModelNotReadyError("model is warming up")
    return {"item_id": item_id, "value": f"item-{item_id}"}


def main() -> None:
    client = TestClient(app)
    ok = client.get("/item/1")
    bad = client.get("/item/-5")
    warm = client.get("/item/999")

    print("ok   :", ok.status_code, ok.json())
    print("bad  :", bad.status_code, bad.json())
    print("warm :", warm.status_code, warm.json())
    print("[Done] 05_error_handling.py completed successfully.")


if __name__ == "__main__":
    main()
