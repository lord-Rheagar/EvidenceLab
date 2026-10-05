"""Probe one uncached review ranking; print only allowlisted rejection metadata."""

import argparse
import json

import typesafe_sdk

from evidencelab.config import config_path, load_config, resolve_services
from evidencelab.evidence import EvidenceError
from evidencelab.reranking import (
    CANDIDATE_LIMIT,
    RERANK_LIMITS,
    JevError,
    candidate_batches,
    json_bytes,
    ranking_key,
    rejection_code,
    rerank,
)
from evidencelab.retrieval import open_index
from evidencelab.storage import open_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--angle", type=int, help="One-based research angle to probe")
    args = parser.parse_args()
    result: dict = {}
    actual = typesafe_sdk.TypeSafeClient

    class ProbeClient(actual):
        def system_one(self, *args, **kwargs):
            state = kwargs["state"]
            result.update(state_bytes=json_bytes(state), question_count=len(kwargs["questions"]))
            try:
                return super().system_one(*args, **kwargs)
            except typesafe_sdk.TypeSafeAPIError as exc:
                result.update(
                    http_status=exc.status,
                    rejection_code=rejection_code(exc.status, exc.body),
                )
                raise

    services = resolve_services(load_config(config_path()))
    with open_store() as store, store.lock(args.run_id):
        job = store.get_job(args.run_id)
        if job.stage != "review" or job.status != "failed" or job.plan is None:
            raise SystemExit("Expected a failed review run; no requests made.")
        if job.rerank_calls >= RERANK_LIMITS[job.profile.depth]:
            raise SystemExit("Jev budget exhausted; no requests made.")
        if args.angle is not None and not 1 <= args.angle <= len(job.plan.angles):
            raise SystemExit("Angle is outside the saved plan; no requests made.")
        angles = job.plan.angles if args.angle is None else [job.plan.angles[args.angle - 1]]
        selected = None
        saved = {p.passage_id: p for p in store.passages(job.run_id)}
        with open_index(services) as index:
            for angle in angles:
                hits = index.search(store.get_run(job.run_id), angle, CANDIDATE_LIMIT)
                if any(saved.get(h.passage.passage_id) != h.passage for h in hits):
                    raise EvidenceError(
                        "Index evidence differs from local snapshots; no Jev request made."
                    )
                for batch in candidate_batches(angle, hits, job.profile.jev_model):
                    key = ranking_key(angle, batch, job.profile.jev_model)
                    if not any(a.cache_key == key for a in job.rerankings):
                        selected = (angle, batch)
                        break
                if selected:
                    break
        if selected is None:
            raise SystemExit("All selected rankings are cached; no Jev request made.")
        angle, hits = selected
        job.rerank_calls += 1
        store.save_job(job)

        def record(event):
            event.stage = job.stage
            event.model_calls = job.model_calls
            event.rerank_calls = job.rerank_calls
            job.diagnostics.append(event)
            job.diagnostics = job.diagnostics[-500:]
            store.save_job(job)

        typesafe_sdk.TypeSafeClient = ProbeClient
        try:
            _, audit = rerank(angle, hits, services, model=job.profile.jev_model, diagnostic=record)
            job.rerankings.append(audit)
            store.save_job(job)
            result["status"] = "passed"
        except JevError as exc:
            result["status"] = exc.code
        finally:
            typesafe_sdk.TypeSafeClient = actual
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
