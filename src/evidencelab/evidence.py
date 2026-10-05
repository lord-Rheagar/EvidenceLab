"""Evidence records shared by extraction, persistence, and retrieval."""

import hashlib
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidencelab.config import Depth

SOURCE_LIMITS = {Depth.QUICK: 10, Depth.BALANCED: 25, Depth.THOROUGH: 50}
MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 300
type SourceKind = Literal["markdown", "text", "pdf", "html"]


class EvidenceError(Exception):
    """A safe-to-display evidence processing or storage failure."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(Record):
    text: str
    locator: str
    page: int | None = None


class Source(Record):
    source_id: str
    run_id: str
    location: str
    aliases: list[str] = Field(default_factory=list)
    title: str
    kind: SourceKind
    content_hash: str
    retrieved_at: str = Field(default_factory=now)
    published_at: str | None = None
    segments: list[Segment]
    warnings: list[str] = Field(default_factory=list)


class Passage(Record):
    passage_id: str
    run_id: str
    source_id: str
    text: str
    title: str
    location: str
    locator: str
    segment: int
    start: int
    end: int
    retrieved_at: str
    published_at: str | None = None


class EvidenceRun(Record):
    run_id: str
    purpose: Literal["evidence", "smoke"] = "evidence"
    created_at: str = Field(default_factory=now)
    depth: Depth = Depth.THOROUGH
    status: Literal["local", "indexing", "indexed", "index_failed"] = "local"
    index_name: str | None = None
    elasticsearch_url: str | None = None
    inference_id: str | None = None


class SearchHit(Record):
    passage: Passage
    score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    rerank_score: float | None = None
    rerank_confidence: float | None = None


def chunk_source(source: Source, *, words: int = 600, overlap: int = 80) -> list[Passage]:
    """Cut exact snapshot substrings, preferring paragraph boundaries near the word limit."""
    if not 0 <= overlap < words:
        raise ValueError("Overlap must be smaller than the positive chunk size")
    passages = []
    for number, segment in enumerate(source.segments):
        tokens = list(re.finditer(r"\S+", segment.text))
        cursor = 0
        while cursor < len(tokens):
            stop = min(cursor + words, len(tokens))
            if stop < len(tokens):
                for candidate in range(stop, cursor + words // 2, -1):
                    gap = segment.text[tokens[candidate - 1].end() : tokens[candidate].start()]
                    if "\n\n" in gap:
                        stop = candidate
                        break
            start, end = tokens[cursor].start(), tokens[stop - 1].end()
            locator = segment.locator
            if segment.page is None:
                first_line = segment.text.count("\n", 0, start) + 1
                last_line = segment.text.count("\n", 0, end) + 1
                locator = f"{locator}, lines {first_line}-{last_line}"
            passages.append(
                Passage(
                    passage_id=digest(source.source_id, str(number), str(start), str(end)),
                    run_id=source.run_id,
                    source_id=source.source_id,
                    text=segment.text[start:end],
                    title=source.title,
                    location=source.location,
                    locator=locator,
                    segment=number,
                    start=start,
                    end=end,
                    retrieved_at=source.retrieved_at,
                    published_at=source.published_at,
                )
            )
            if stop == len(tokens):
                break
            cursor = max(cursor + 1, stop - overlap)
    return passages
