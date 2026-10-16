"""FastAPI service: JSON API at /predict, Gradio UI at /, docs at /docs."""
import time
from collections import defaultdict, deque

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from demo import build_demo
from predict import CLASSES, predict_text

app = FastAPI(title="Hate Speech Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)

# --- tiny per-IP rate limiter (30 requests / minute) for the JSON API ---
RATE_LIMIT, WINDOW = 30, 60
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # set by Nginx
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    ip, now = _client_ip(request), time.time()
    q = _hits[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    q.append(now)
    try:
        return predict_text(req.text)  # input text is never logged or stored
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# Mounted last so /health, /predict and /docs take priority over the UI at "/".
app = gr.mount_gradio_app(app, build_demo(), path="/")
