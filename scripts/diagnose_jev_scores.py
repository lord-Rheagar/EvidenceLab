"""One explicitly budgeted Jev probe; retain numeric diagnostics, never request text/keys."""

import argparse
import json
import math
from pathlib import Path

import typesafe_sdk

from evidencelab.config import config_path, load_config, resolve_services
from evidencelab.reranking import CANDIDATE_LIMIT, RERANK_LIMITS, JevError, rerank
from evidencelab.retrieval import open_index
from evidencelab.storage import EvidenceStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Diagnostic output already exists; no requests made.")
    result = {"run_id": args.run_id, "answers": []}
    actual = typesafe_sdk.TypeSafeClient

    class DiagnosticClient(actual):
        def system_one(self, *args, **kwargs):
            response = super().system_one(*args, **kwargs)
            result["model"] = response.model
            for name, answer in response.answers.items():
                expected = sum(level * p for level, p in answer.probabilities.items())
                result["answers"].append(
                    {
                        "candidate": name,
                        "score": answer.score,
                        "probabilities": answer.probabilities,
                        "weighted_sum": expected,
                        "probability_sum": sum(answer.probabilities.values()),
                        "difference": abs(answer.score - expected),
                        "passes_current_check": math.isclose(
                            answer.score, expected, rel_tol=0, abs_tol=0.02 + 1e-12
                        ),
                    }
                )
            return response

    settings = resolve_services(load_config(config_path()))
    store = EvidenceStore()
    try:
        with store.lock(args.run_id):
            job = store.get_job(args.run_id)
            assert job.plan is not None and job.stage == "review" and job.status == "failed"
            if job.rerank_calls >= RERANK_LIMITS[job.profile.depth]:
                raise SystemExit("Saved Jev budget exhausted; no requests made.")
            with open_index(settings) as index:
                hits = index.search(store.get_run(job.run_id), job.plan.angles[0], CANDIDATE_LIMIT)
            job.rerank_calls += 1
            store.save_job(job)
            typesafe_sdk.TypeSafeClient = DiagnosticClient
            try:
                _, audit = rerank(job.plan.angles[0], hits, settings, model=job.profile.jev_model)
                job.rerankings.append(audit)
                store.save_job(job)
                result["status"] = "passed"
            except JevError as exc:
                result["status"] = exc.code
            finally:
                typesafe_sdk.TypeSafeClient = actual
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
