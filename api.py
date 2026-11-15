from __future__ import annotations

from fastapi import FastAPI, HTTPException

from service.predict_service import explain_record, list_records, predict_record

app = FastAPI(title="铝合金屈服强度预测服务", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "yield-strength-prediction"}


@app.post("/predict")
def predict(payload: dict) -> dict:
    try:
        return predict_record(payload, save=True)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/explain")
def explain(payload: dict) -> dict:
    try:
        return explain_record(payload)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/records")
def records(limit: int = 200) -> list[dict]:
    return list_records(limit=limit)
