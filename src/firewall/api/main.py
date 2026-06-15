from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import HTMLResponse, PlainTextResponse

from firewall.api.schemas import (
    AnalyticsResponse,
    ChatRequest,
    ChatResponse,
    InspectRequest,
    InspectResponse,
)
from firewall.service import FirewallService

REQUESTS_TOTAL = Counter(
    "firewall_requests_total",
    "Total firewall requests",
    ["endpoint", "decision"],
)
LATENCY = Histogram(
    "firewall_latency_seconds",
    "Firewall request latency",
    ["stage"],
)

service: FirewallService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = FirewallService()
    yield


app = FastAPI(
    title="Prompt Firewall API",
    description="Runtime security gateway for LLM chatbots",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Prompt Firewall</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
    h1 { margin-bottom: 0.25rem; }
    p { color: #444; }
    ul { line-height: 1.8; }
    a { color: #2563eb; }
    code { background: #f3f4f6; padding: 0.15rem 0.4rem; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Prompt Firewall API</h1>
  <p>Runtime security gateway — rule engine + DistilBERT + Groq LLaMA 3.1 8B.</p>
  <ul>
    <li><a href="/docs">Interactive API docs</a> (try <code>POST /inspect</code> and <code>POST /chat</code>)</li>
    <li><a href="/health">Health check</a></li>
    <li><a href="/analytics">Analytics</a></li>
    <li><a href="/metrics">Prometheus metrics</a></li>
  </ul>
  <p>Example: <code>POST /inspect</code> with body <code>{"prompt": "What is the capital of France?"}</code></p>
</body>
</html>"""


@app.get("/health")
async def health() -> dict[str, str | bool]:
    assert service is not None
    return {
        "status": "ok",
        "classifier_loaded": service.classifier.is_loaded,
        "llm_provider": service.config.llm.get("provider", "groq"),
        "llm_model": service.config.llm.get("model", ""),
        "model_version": service.config.model_version,
    }


@app.post("/inspect", response_model=InspectResponse)
async def inspect(request: InspectRequest) -> InspectResponse:
    assert service is not None
    result = service.inspect(request.prompt)
    REQUESTS_TOTAL.labels(endpoint="inspect", decision=result.action).inc()
    for stage, ms in result.latency_ms.items():
        LATENCY.labels(stage=stage).observe(ms / 1000.0)
    return InspectResponse(**result.__dict__)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    assert service is not None
    result = await service.chat(request.prompt)
    REQUESTS_TOTAL.labels(endpoint="chat", decision=result.action).inc()
    for stage, ms in result.latency_ms.items():
        LATENCY.labels(stage=stage).observe(ms / 1000.0)
    return ChatResponse(**result.__dict__)


@app.get("/analytics", response_model=AnalyticsResponse)
async def analytics() -> AnalyticsResponse:
    assert service is not None
    summary = service.telemetry.analytics_summary()
    return AnalyticsResponse(**summary)


@app.get("/events")
async def events(
    limit: int = Query(50, ge=1, le=500),
    decision: str | None = None,
) -> list[dict]:
    assert service is not None
    return service.telemetry.query_events(limit=limit, decision=decision)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type="text/plain")
