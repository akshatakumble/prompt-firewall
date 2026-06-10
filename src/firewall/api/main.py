from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
