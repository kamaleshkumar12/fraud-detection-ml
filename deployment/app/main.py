import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import Field, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MAX_BATCH = 1000
state = {}

fields = {"Time": (float, Field(..., ge=0)), "Amount": (float, Field(..., ge=0))}
for i in range(1, 29):
    fields[f"V{i}"] = (float, ...)
Transaction = create_model("Transaction", **fields)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(os.path.join(MODEL_DIR, "fraud_model.joblib"))
    state["scaler"] = joblib.load(os.path.join(MODEL_DIR, "robust_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        state["config"] = json.load(f)
    logger.info("Model loaded: %s", state["config"]["model_name"])
    yield
    state.clear()


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


def risk_level(p: float) -> str:
    if p >= 0.80:
        return "CRITICAL"
    if p >= 0.60:
        return "HIGH"
    if p >= 0.30:
        return "MEDIUM"
    return "LOW"


def score(records: list) -> list:
    cfg = state["config"]
    raw = pd.DataFrame(records)
    raw["Hour"] = ((raw["Time"] // 3600) % 24).astype(int)
    raw["Amount_log"] = np.log1p(raw["Amount"])
    X = raw[cfg["features"]]
    Xs = pd.DataFrame(state["scaler"].transform(X), columns=cfg["features"])
    probs = state["model"].predict_proba(Xs)[:, 1]
    results = []
    for p in probs:
        p = float(p)
        results.append({
            "fraud_probability": round(p, 4),
            "risk_score": round(p * 100, 2),
            "decision": "FRAUD" if p >= cfg["threshold"] else "LEGITIMATE",
            "risk_level": risk_level(p),
        })
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    cfg = state["config"]
    return {"model_name": cfg["model_name"], "threshold": cfg["threshold"],
            "n_features": len(cfg["features"])}


@app.post("/predict")
def predict(tx: Transaction):
    return score([tx.model_dump()])[0]


@app.post("/predict/batch")
def predict_batch(txs: List[Transaction]):
    if len(txs) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Max {MAX_BATCH} transactions per request")
    results = score([t.model_dump() for t in txs])
    return {"count": len(results), "results": results}
