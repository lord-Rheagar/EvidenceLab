"""Explicit, paid CLI evaluation cases. Never retry or overwrite prior attempts."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from evidencelab.research import context_passages, validate_claims
from evidencelab.storage import EvidenceStore

CASES = [
    (
        "postgres-indexes",
        "quick",
        "When should PostgreSQL JSONB queries use a GIN index rather than a B-tree "
        "expression index? Explain supported operators, write overhead, and cases where "
        "an index does not help. Prefer official PostgreSQL documentation.",
        ["--web"],
    ),
    (
        "python-concurrency",
        "quick",
        "In Python, when should blocking I/O be run with asyncio.to_thread instead of "
        "calling it directly inside a coroutine? Explain cancellation and CPU-bound "
        "limitations using only the supplied documentation; identify missing evidence.",
        [
            "--no-web",
            "--url",
            "https://docs.python.org/3/library/asyncio-task.html",
            "--url",
            "https://docs.python.org/3/library/concurrent.futures.html",
        ],
    ),
    (
        "redis-locks",
        "balanced",
        "Can a Redis SET NX lock with an expiry guarantee mutual exclusion for writes "
        "to an external database during pauses and network failures? Explain ownership, "
        "safe release, lease expiry, and fencing. Distinguish documented guarantees "
        "from assumptions and prefer primary sources.",
        ["--web"],
    ),
    (
        "elastic-hybrid",
        "balanced",
        "For an Elasticsearch technical-documentation search service, when is keyword "
        "search preferable to semantic search, and when should hybrid RRF retrieval "
        "be used? Discuss exact identifiers, filters, evaluation, and operational costs. "
        "Prefer official documentation and clearly qualify version-specific claims.",
        ["--web"],
    ),
    (
        "kafka-guarantees",
        "thorough",
        "Do Kafka exactly-once processing guarantees extend automatically to an "
        "external PostgreSQL database or HTTP API? Explain producer idempotence, "
        "transactions, consumer isolation, retries, and practical end-to-end designs. "
        "Use primary sources and distinguish guarantees from application responsibilities.",
        ["--web"],
    ),
    (
        "pgvector-recall",
        "thorough",
        "A team claims pgvector HNSW always guarantees 100 percent recall and beats "
        "IVFFlat at every dataset size. Is this supported? Compare build cost, memory, "
        "filtered-query behavior, tuning, and when exact search is appropriate. "
        "Prefer official pgvector documentation and papers; do not invent benchmarks.",
        ["--web"],
    ),
]


def cli(args: list[str], output: Path, timeout: int = 1200) -> int:
    with output.open("x", encoding="utf-8") as stream:
        return subprocess.run(
            [sys.executable, "-c", "from evidencelab.cli import app; app()", *args],
            stdout=stream,
            timeout=timeout,
            check=False,
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(1, len(CASES) + 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume", help="Explicitly resume a known failed run instead of creating one"
    )
    args = parser.parse_args()
    name, depth, question, options = CASES[args.case - 1]
    directory = args.output_dir / f"{args.case:02d}-{name}"
    directory.mkdir(parents=True, exist_ok=False)
    command = (
        ["resume", args.resume, "--json"]
        if args.resume
        else ["research", question, "--depth", depth, *options, "--json"]
    )
    print(f"START case={args.case} depth={depth} name={name}", flush=True)
    started = time.monotonic()
    code = cli(command, directory / "result.json")
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    identifier = payload.get("research", payload)["run_id"]
    store = EvidenceStore()
    try:
        job = store.get_job(identifier)
        result = {
            "case": args.case,
            "name": name,
            "depth": depth,
            "run_id": identifier,
            "status": job.status,
            "stage": job.stage,
            "error": job.error,
            "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "model": job.profile.model,
            "reranker": job.profile.reranker.value,
            "model_calls": job.model_calls,
            "search_calls": job.search_calls,
            "rerank_calls": job.rerank_calls,
            "gap_passes": job.gap_passes,
            "source_count": len(store.sources(identifier)),
            "passage_count": len(store.passages(identifier)),
            "warnings": job.warnings,
        }
        if job.status == "complete":
            assert code == 0 and job.report is not None
            evidence = context_passages(store, job)
            claims = [*job.report.summary, *job.report.findings, *job.report.tradeoffs]
            validate_claims(claims, evidence)
            result["citation_integrity"] = "pass"
            result["claim_count"] = len(claims)
            result["citation_count"] = sum(len(c.citations) for c in claims)
            result["cited_domains"] = sorted(
                {
                    urlsplit(evidence[c.passage_id].location).netloc
                    for claim in claims
                    for c in claim.citations
                }
            )
            for format, filename in [("markdown", "report.md"), ("json", "report.json")]:
                assert (
                    cli(
                        [
                            "export",
                            identifier,
                            "--format",
                            format,
                            "--output",
                            str(directory / filename),
                        ],
                        directory / f"export-{format}.log",
                    )
                    == 0
                )
            assert cli(["resume", identifier, "--json"], directory / "completed-resume.json") == 0
            assert store.get_job(identifier) == job
            assert (
                json.loads((directory / "completed-resume.json").read_text(encoding="utf-8"))
                == payload
            )
            result["exports_and_completed_resume"] = "pass"
        (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
