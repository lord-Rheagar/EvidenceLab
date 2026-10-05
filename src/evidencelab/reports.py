"""Deterministic rendering and offline export of validated research reports."""

import json
import re
from pathlib import Path
from urllib.parse import quote

from evidencelab.evidence import EvidenceError
from evidencelab.research import context_passages, validate_claims
from evidencelab.research_records import Citation, ResearchJob
from evidencelab.storage import EvidenceStore


def plain(value: str) -> str:
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    return re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", value)


def render_report(store: EvidenceStore, job: ResearchJob) -> str:
    if job.status != "complete" or job.report is None:
        raise EvidenceError("Research report is not complete. Resume the run first.")
    report = job.report
    evidence = context_passages(store, job)
    validate_claims([*report.summary, *report.findings, *report.tradeoffs], evidence)
    jev_versions = plain(", ".join(sorted({a.model for a in job.rerankings})) or "none")
    lines = [
        f"# {plain(report.title)}",
        "",
        f"Question: {plain(job.question)}",
        "",
        f"Run: `{job.run_id}`",
        "",
        f"Model: {plain(job.profile.provider.value)} / {plain(job.profile.model)}",
        "",
        f"Depth: {job.profile.depth.value}; web discovery: {'enabled' if job.web else 'disabled'}; "
        f"model requests: {job.model_calls}; web searches: {job.search_calls}.",
        "",
        f"Reranker: {job.profile.reranker.value}; Jev requests: {job.rerank_calls}; "
        f"returned Jev versions: {jev_versions}.",
        "",
        "Scope: up to four retrieved passages per angle, 4,000 characters per passage, "
        "48,000 evidence characters per review. Not an exhaustive source audit.",
        "",
        "Citation IDs and quotations checked against saved evidence. "
        "Claim entailment and source reliability require human review.",
        "",
    ]
    if job.rerankings:
        windows = " / ".join(
            f"{value:,}" for value in sorted({a.excerpt_chars for a in job.rerankings})
        )
        lines.extend(
            [
                "Jev ranks up to 12 candidates per angle; "
                f"recorded excerpt limits: {windows} characters. "
                "Relevance scores and confidence are not factual verification.",
                "",
            ]
        )
    citations: list[Citation] = []
    for heading, claims in [
        ("Summary", report.summary),
        ("Findings", report.findings),
        ("Tradeoffs", report.tradeoffs),
    ]:
        if not claims:
            continue
        lines.extend([f"## {heading}", ""])
        for claim in claims:
            labels = []
            for citation in claim.citations:
                if citation not in citations:
                    citations.append(citation)
                labels.append(f"[{citations.index(citation) + 1}]")
            lines.extend([f"- {plain(claim.text)} {' '.join(labels)}", ""])
    lines.extend(["## Limitations", ""])
    limitations = list(
        dict.fromkeys(
            [
                *report.limitations,
                *(job.review.gaps if job.review else []),
                *job.warnings,
            ]
        )
    )
    for limitation in limitations:
        lines.append(f"- {plain(limitation)}")
    lines.extend(["", "## Evidence", ""])
    for number, citation in enumerate(citations, 1):
        p = evidence[citation.passage_id]
        location = quote(p.location, safe=":/?=&%#@+-._~")
        lines.extend(
            [
                f"[{number}] [{plain(p.title)}](<{location}>)",
                "",
                f"{plain(p.locator)}; retrieved {plain(p.retrieved_at)}; "
                f"published {plain(p.published_at or 'unknown')}",
                "",
                f"> {plain(citation.quote)}",
                "",
                f"Passage: `{p.passage_id}`",
                "",
            ]
        )
    return "\n".join(lines)


def export_report(store: EvidenceStore, job: ResearchJob, output: Path, format: str) -> None:
    markdown = render_report(store, job)
    if format == "markdown":
        content = markdown
    elif format == "json":
        content = json.dumps(
            {
                "schema_version": 1,
                "research": job.model_dump(mode="json"),
                "evidence": [
                    p.model_dump(mode="json") for p in context_passages(store, job).values()
                ],
            },
            indent=2,
        )
    else:
        raise EvidenceError("Export format must be markdown or json.")
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    except FileExistsError:
        raise EvidenceError(
            "Output already exists; choose a new path. Existing files are never overwritten."
        ) from None
    except OSError:
        raise EvidenceError(
            "Cannot write export; check the output directory and permissions."
        ) from None
