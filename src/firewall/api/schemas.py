from __future__ import annotations

from pydantic import BaseModel, Field


class InspectRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)


class InspectResponse(BaseModel):
    request_id: str
    action: str
    risk_score: float
    triggered_rules: list[str]
    sanitized_prompt: str | None
    reason: str
    is_near_miss: bool
    latency_ms: dict[str, float]


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    request_id: str
    action: str
    response: str
    output_status: str
    risk_score: float
    triggered_rules: list[str]
    violation_reasons: list[str]
    latency_ms: dict[str, float]


class AnalyticsResponse(BaseModel):
    total_requests: int
    decisions: dict[str, int]
    avg_risk_score: float
    top_rules: list[dict[str, int | str]]
    near_misses: int
    avg_latency_ms: dict[str, float]
