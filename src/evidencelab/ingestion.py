"""Bounded ingestion into a run's immutable local evidence snapshots."""

from pathlib import Path

from evidencelab.config import Depth
from evidencelab.documents import expand_files, read_file, read_url
from evidencelab.evidence import SOURCE_LIMITS, EvidenceError
from evidencelab.fetching import normalize_url
from evidencelab.storage import EvidenceStore


def ingest_sources(
    store: EvidenceStore,
    files: list[Path],
    urls: list[str],
    *,
    run_id: str | None = None,
    depth: Depth = Depth.THOROUGH,
) -> dict[str, object]:
    if not files and not urls:
        raise EvidenceError("Supply at least one --file or --url.")
    run = store.get_run(run_id) if run_id else None
    limit = SOURCE_LIMITS[run.depth if run else depth]
    paths, skipped = expand_files(files, limit)
    normalized_urls = list(dict.fromkeys(normalize_url(url) for url in urls))
    if len(paths) + len(normalized_urls) > limit:
        raise EvidenceError(f"Input set exceeds the {limit}-source limit.")
    if run is None:
        run = store.create_run(depth)
    with store.lock(run.run_id):
        existing = store.sources(run.run_id)
        known = {loc for source in existing for loc in [source.location, *source.aliases]}
        candidates = [path.as_uri() for path in paths] + normalized_urls
        if len(existing) + len(set(candidates) - known) > limit:
            raise EvidenceError(
                f"Adding these inputs would exceed this run's {limit}-source limit."
            )
        added, duplicates = 0, 0
        inputs: list[Path | str] = [*paths, *normalized_urls]
        for candidate in inputs:
            location = candidate.as_uri() if isinstance(candidate, Path) else candidate
            if location in known:
                duplicates += 1
                continue
            try:
                source = (
                    read_file(run.run_id, candidate)
                    if isinstance(candidate, Path)
                    else read_url(run.run_id, candidate)
                )
                if store.add_source(source):
                    added += 1
                else:
                    duplicates += 1
                known.update([source.location, *source.aliases])
            except EvidenceError as exc:
                skipped.append({"input": str(candidate), "reason": str(exc)})
        return {
            "run_id": run.run_id,
            "added_sources": added,
            "duplicates": duplicates,
            "sources": len(store.sources(run.run_id)),
            "passages": len(store.passages(run.run_id)),
            "skipped": skipped,
            "warnings": [
                warning for source in store.sources(run.run_id) for warning in source.warnings
            ],
        }
