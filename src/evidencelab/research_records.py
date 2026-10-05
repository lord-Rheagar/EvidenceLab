"""Bounded research contracts and persisted, non-secret run settings."""

from typing import Annotated, Literal

from pydantic import Field, create_model

from evidencelab.config import Depth, ModelProfile
from evidencelab.diagnostics import Diagnostic
from evidencelab.evidence import Record
from evidencelab.reranking import RerankAudit

type ShortText = Annotated[str, Field(min_length=1, max_length=1000)]
type Stage = Literal["plan", "gather", "index", "review", "write", "complete"]

LIMITS = {
    Depth.QUICK: (2, 3, 6, 0, 30),
    Depth.BALANCED: (3, 4, 12, 1, 60),
    Depth.THOROUGH: (4, 6, 24, 2, 120),
}


class ResearchPlan(Record):
    """Distinct technical research angles and targeted public-web queries."""

    angles: list[ShortText] = Field(min_length=2, max_length=6)
    queries: list[ShortText] = Field(max_length=12)


def plan_schema(depth: Depth, *, web: bool) -> type[ResearchPlan]:
    """Expose the actual run limits to generation, not only to post-call checks."""
    low, high, searches, _, _ = LIMITS[depth]
    return create_model(
        "ResearchPlan",
        __base__=ResearchPlan,
        angles=(list[ShortText], Field(min_length=low, max_length=high)),
        queries=(list[ShortText], Field(max_length=min(searches, high * 2) if web else 0)),
    )


class Citation(Record):
    passage_id: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Copy the complete passage_id from the supplied evidence; never abbreviate it.",
    )
    quote: str = Field(
        min_length=12,
        max_length=2000,
        description=(
            "Copy a short, contiguous excerpt verbatim from that passage's text. "
            "Preserve spelling, punctuation and whitespace; no paraphrases or added ellipses."
        ),
    )


class Claim(Record):
    text: str = Field(min_length=1, max_length=2000)
    citations: list[Citation] = Field(min_length=1, max_length=4)


class EvidenceReview(Record):
    """Evidence-backed findings, limitations, and queries for missing evidence."""

    findings: list[Claim] = Field(max_length=16)
    gaps: list[ShortText] = Field(max_length=8)
    follow_up_queries: list[ShortText] = Field(max_length=6)


class ResearchReport(Record):
    """All factual prose must be in cited claims; uncertainty belongs in limitations."""

    title: ShortText
    summary: list[Claim] = Field(min_length=1, max_length=4)
    findings: list[Claim] = Field(min_length=1, max_length=16)
    tradeoffs: list[Claim] = Field(max_length=8)
    limitations: list[ShortText] = Field(min_length=1, max_length=12)


class ResearchJob(Record):
    schema_version: Literal[1] = 1
    run_id: str
    question: str = Field(min_length=1, max_length=8000)
    profile_name: str
    profile: ModelProfile
    elasticsearch_url: str
    inference_id: str
    web: bool
    files: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    stage: Stage = "plan"
    status: Literal["pending", "running", "failed", "interrupted", "complete"] = "pending"
    error: str | None = None
    plan: ResearchPlan | None = None
    pending_queries: list[str] = Field(default_factory=list)
    searched_queries: list[str] = Field(default_factory=list)
    attempted_locations: list[str] = Field(default_factory=list)
    model_calls: int = 0
    search_calls: int = 0
    rerank_calls: int = 0
    rerankings: list[RerankAudit] = Field(default_factory=list)
    gap_passes: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    review_snapshot_key: str | None = None
    review: EvidenceReview | None = None
    report: ResearchReport | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list, max_length=500)
