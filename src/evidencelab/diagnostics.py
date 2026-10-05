"""Allowlisted operational metadata. Never serialize exceptions or response bodies."""

from datetime import UTC, datetime
from typing import Literal

import httpx
import httpx2
from pydantic import Field, ValidationError, field_validator

from evidencelab.evidence import Record

CODES = frozenset(
    {
        "ok",
        "unknown",
        "authentication",
        "access_denied",
        "quota",
        "rate_limit",
        "timeout",
        "connection",
        "request_rejected",
        "context_limit",
        "provider_unavailable",
        "invalid_response",
        "candidate_mismatch",
        "rubric_mismatch",
        "invalid_probabilities",
        "score_mismatch",
        "wrong_order",
        "schema_invalid",
        "output_truncated",
        "missing_tool_call",
        "invalid_tool_arguments",
        "unexpected_tool",
        "stage_failed",
        "interrupted",
        "content_filter",
    }
)

type FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "other"]


class Diagnostic(Record):
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    service: Literal["model", "jev", "pipeline"]
    operation: Literal["request", "generation", "stage"] = "request"
    code: str
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    stage: Literal["plan", "gather", "index", "review", "write", "complete"] | None = None
    model_calls: int = Field(default=0, ge=0)
    rerank_calls: int = Field(default=0, ge=0)
    http_status: int | None = Field(default=None, ge=100, le=599, strict=True)
    schema_name: Literal["ResearchPlan", "EvidenceReview", "ResearchReport"] | None = None
    finish_reason: FinishReason | None = None
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    tool_calls: int | None = Field(default=None, ge=0)
    invalid_tool_calls: int | None = Field(default=None, ge=0)
    candidate_index: int | None = Field(default=None, ge=0, le=11)
    candidate_count: int | None = Field(default=None, ge=1, le=12)
    state_bytes: int | None = Field(default=None, ge=0)
    request_bytes: int | None = Field(default=None, ge=0)
    score: float | None = Field(default=None, allow_inf_nan=False)
    probabilities: list[float] | None = Field(default=None, min_length=4, max_length=4)
    weighted_score: float | None = Field(default=None, allow_inf_nan=False)
    probability_sum: float | None = Field(default=None, allow_inf_nan=False)
    rounding_matches: int = Field(default=0, ge=0, le=12)
    validation_details: str | None = Field(default=None, max_length=2000)

    @field_validator("code")
    @classmethod
    def known_code(cls, value: str) -> str:
        if value not in CODES:
            raise ValueError("Unknown diagnostic category")
        return value

    @field_validator("probabilities")
    @classmethod
    def finite_probabilities(cls, value: list[float] | None) -> list[float] | None:
        import math

        if value is not None and any(not math.isfinite(p) for p in value):
            raise ValueError("Non-finite diagnostic probabilities")
        return value


def classify_exception(exc: BaseException) -> tuple[str, int | None]:
    """Inspect only exception types and numeric statuses, including SDK wrappers."""
    from openrouter.errors import ResponseValidationError

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if type(status) is not int:
            status = getattr(current, "status", None)
        if type(status) is int and 100 <= status <= 599:
            if isinstance(current, ResponseValidationError):
                return "invalid_response", status
            code = {
                401: "authentication",
                403: "access_denied",
                402: "quota",
                429: "rate_limit",
                400: "request_rejected",
                404: "request_rejected",
                422: "request_rejected",
                408: "timeout",
                504: "timeout",
            }.get(status, "provider_unavailable" if status >= 500 else "unknown")
            return code, status
        if isinstance(current, (TimeoutError, httpx.TimeoutException, httpx2.TimeoutException)):
            return "timeout", None
        if isinstance(current, (httpx.NetworkError, httpx2.NetworkError, ConnectionError)):
            return "connection", None
        if isinstance(current, (ValidationError, ValueError, KeyError, TypeError)):
            # Prefer a typed transport cause to a generic SDK parsing wrapper.
            cause = current.__cause__ or current.__context__
            if cause is not None and id(cause) not in seen:
                current = cause
                continue
            return "invalid_response", None
        current = current.__cause__ or current.__context__
    return "unknown", None
