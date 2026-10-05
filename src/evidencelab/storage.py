"""Local evidence snapshots and run metadata, outside the repository."""

import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout
from platformdirs import user_data_path

from evidencelab.config import Depth
from evidencelab.evidence import (
    SOURCE_LIMITS,
    EvidenceError,
    EvidenceRun,
    Passage,
    Source,
    chunk_source,
)
from evidencelab.research_records import ResearchJob


def data_path() -> Path:
    value = os.getenv("EVIDENCE_DATA_DIR")
    return Path(value).expanduser() if value else user_data_path("evidencelab")


class EvidenceStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_path()
        self.path.mkdir(parents=True, exist_ok=True)
        # LangGraph runs sequential nodes on worker threads; the run lock owns writes.
        self.connection = sqlite3.connect(
            self.path / "evidence.sqlite3", timeout=10, check_same_thread=False
        )
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS evidence_runs (
                run_id TEXT PRIMARY KEY, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES evidence_runs(run_id),
                location TEXT NOT NULL, content_hash TEXT NOT NULL, data TEXT NOT NULL,
                UNIQUE(run_id, location), UNIQUE(run_id, content_hash)
            );
            CREATE TABLE IF NOT EXISTS passages (
                passage_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES evidence_runs(run_id),
                source_id TEXT NOT NULL REFERENCES sources(source_id), data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS passages_run ON passages(run_id);
            CREATE TABLE IF NOT EXISTS research_jobs (
                run_id TEXT PRIMARY KEY REFERENCES evidence_runs(run_id), data TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def save_job(self, job: ResearchJob) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO research_jobs VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET data=excluded.data",
                (job.run_id, job.model_dump_json()),
            )

    def get_job(self, run_id: str) -> ResearchJob:
        row = self.connection.execute(
            "SELECT data FROM research_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise EvidenceError("No research job for this run. Start with 'evidence research'.")
        return ResearchJob.model_validate_json(row[0])

    def job_summary(self, run_id: str) -> dict[str, object] | None:
        exists = self.connection.execute(
            "SELECT 1 FROM research_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not exists:
            return None
        job = self.get_job(run_id)
        return {
            "question": job.question,
            "status": job.status,
            "stage": job.stage,
            "profile": job.profile_name,
            "provider": job.profile.provider.value,
            "model": job.profile.model,
            "model_calls": job.model_calls,
            "search_calls": job.search_calls,
            "rerank_calls": job.rerank_calls,
            "gap_passes": job.gap_passes,
            "warnings": job.warnings,
            "error": job.error,
            "diagnostics": [event.model_dump(mode="json") for event in job.diagnostics],
        }

    @contextmanager
    def lock(self, run_id: str) -> Iterator[None]:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise EvidenceError("Invalid run ID.")
        try:
            with FileLock(self.path / f"{run_id}.lock", timeout=0):
                yield
        except Timeout:
            raise EvidenceError(
                "Another process is updating this run; retry when it finishes."
            ) from None

    def create_run(self, depth: Depth) -> EvidenceRun:
        run = EvidenceRun(run_id=uuid4().hex, depth=depth)
        self.save_run(run)
        return run

    def save_run(self, run: EvidenceRun) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO evidence_runs(run_id, data) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET data=excluded.data",
                (run.run_id, run.model_dump_json()),
            )

    def get_run(self, run_id: str) -> EvidenceRun:
        row = self.connection.execute(
            "SELECT data FROM evidence_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise EvidenceError("Run not found. Use 'evidence runs' to see saved runs.")
        return EvidenceRun.model_validate_json(row[0])

    def runs(self) -> list[EvidenceRun]:
        return [
            EvidenceRun.model_validate_json(row[0])
            for row in self.connection.execute("SELECT data FROM evidence_runs ORDER BY rowid DESC")
        ]

    def sources(self, run_id: str) -> list[Source]:
        self.get_run(run_id)
        return [
            Source.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT data FROM sources WHERE run_id = ? ORDER BY rowid", (run_id,)
            )
        ]

    def passages(self, run_id: str) -> list[Passage]:
        self.get_run(run_id)
        return [
            Passage.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT data FROM passages WHERE run_id = ? ORDER BY rowid", (run_id,)
            )
        ]

    def add_source(self, source: Source) -> bool:
        run = self.get_run(source.run_id)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            sources = self.sources(source.run_id)
            for existing in sources:
                locations = {existing.location, *existing.aliases}
                if source.location in locations or existing.content_hash == source.content_hash:
                    existing.aliases = sorted(
                        (locations | {source.location, *source.aliases}) - {existing.location}
                    )
                    self.connection.execute(
                        "UPDATE sources SET data = ? WHERE source_id = ?",
                        (existing.model_dump_json(), existing.source_id),
                    )
                    return False
            if len(sources) >= SOURCE_LIMITS[run.depth]:
                raise EvidenceError("This run has reached its source limit; start another run.")
            self.connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (
                    source.source_id,
                    source.run_id,
                    source.location,
                    source.content_hash,
                    source.model_dump_json(),
                ),
            )
            self.connection.executemany(
                "INSERT INTO passages VALUES (?, ?, ?, ?)",
                [
                    (p.passage_id, p.run_id, p.source_id, p.model_dump_json())
                    for p in chunk_source(source)
                ],
            )
            run.status = "local"
            self.connection.execute(
                "UPDATE evidence_runs SET data = ? WHERE run_id = ?",
                (run.model_dump_json(), run.run_id),
            )
        return True


@contextmanager
def open_store() -> Iterator[EvidenceStore]:
    store = None
    try:
        store = EvidenceStore()
        yield store
    except (OSError, sqlite3.Error):
        raise EvidenceError(
            "Local evidence storage failed; check disk access and available space."
        ) from None
    finally:
        if store:
            store.close()
