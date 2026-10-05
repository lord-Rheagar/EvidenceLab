"""Explicitly invoked, small live checks using synthetic evidence only."""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from evidencelab.config import Depth, ModelProfile, Reranker, Services
from evidencelab.documents import extract_source
from evidencelab.evidence import EvidenceError, SearchHit
from evidencelab.providers import PROBE_ERRORS, ModelProbeError, build_model, check_model
from evidencelab.reranking import JEV_ERRORS, JevError
from evidencelab.research import (
    discover,
    evidence_context,
    validate_claims,
    validate_research_settings,
)
from evidencelab.research_model import ResearchModel
from evidencelab.research_records import Citation, Claim
from evidencelab.retrieval import open_index
from evidencelab.storage import EvidenceStore

EXPECTED_RETRY_LIMIT = 17
EXPECTED_SENTENCE = f"The smoke_retry_limit setting is exactly {EXPECTED_RETRY_LIMIT} attempts."


class SmokeAnswer(BaseModel):
    """Extract the integer retry limit and cite its exact supporting sentence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    retry_limit: int
    citations: list[Citation] = Field(min_length=1, max_length=4)


def smoke_test(
    store: EvidenceStore,
    name: str,
    profile: ModelProfile,
    services: Services,
    web: bool,
    progress: Callable[[str], None],
) -> dict[str, object]:
    validate_research_settings(profile, services, web)
    model = build_model(name, profile)  # Credential presence before paid requests or writes.
    run = store.create_run(Depth.QUICK)
    run.purpose = "smoke"
    store.save_run(run)
    checks = dict.fromkeys(
        ["model_probe", "index", "retrieval", "cited_answer", "tavily", "jev"], "not_run"
    )
    checks["tavily"] = "not_run" if web else "skipped"
    checks["jev"] = "not_run" if profile.reranker == Reranker.JEV else "skipped"
    result: dict[str, object] = {
        "status": "running",
        "run_id": run.run_id,
        "profile": name,
        "provider": profile.provider.value,
        "model": profile.model,
        "checks": checks,
    }
    stage = "model_probe"
    try:
        with store.lock(run.run_id):
            progress("Smoke: model tool-call probe (paid request)")
            check_model(name, profile)
            checks[stage] = "passed"
            stage = "index"
            relevant = extract_source(
                run.run_id,
                "smoke://fixture/retry",
                f"Synthetic EvidenceLab test fixture. {EXPECTED_SENTENCE}".encode(),
                "text",
                "Smoke fixture",
            )
            unrelated = extract_source(
                run.run_id,
                "smoke://fixture/colors",
                b"Synthetic EvidenceLab test fixture. "
                b"The interface color options are green and red.",
                "text",
                "Unrelated fixture",
            )
            store.add_source(relevant)
            store.add_source(unrelated)
            question = "What is the smoke_retry_limit setting in the synthetic fixture?"
            with open_index(services) as index:
                progress("Smoke: index two synthetic passages and run hybrid retrieval")
                index.index_run(store, run.run_id)
                checks[stage] = "passed"
                stage = "retrieval"
                hits = index.search(store.get_run(run.run_id), question, 2)
            saved = {p.passage_id: p for p in store.passages(run.run_id)}
            if not hits or any(saved.get(h.passage.passage_id) != h.passage for h in hits):
                raise EvidenceError("Smoke retrieval failed snapshot integrity validation.")
            if not any(h.passage.source_id == relevant.source_id for h in hits):
                raise EvidenceError("Smoke retrieval did not find the expected fixture.")
            checks[stage] = "passed"
            if profile.reranker == Reranker.JEV:
                from evidencelab.reranking import rerank

                stage = "jev"
                progress("Smoke: one Jev relevance request")
                # Always test both fixtures, with the irrelevant one first to reject ties.
                candidates = sorted(
                    [SearchHit(passage=p, score=0) for p in saved.values()],
                    key=lambda h: h.passage.source_id == relevant.source_id,
                )
                hits, audit = rerank(question, candidates, services, model=profile.jev_model)
                result["jev_model"] = audit.model
                if hits[0].passage.source_id != relevant.source_id:
                    raise JevError("wrong_order")
                checks[stage] = "passed"
            stage = "cited_answer"
            progress("Smoke: one structured, cited answer request")
            calls = 0

            def reserve() -> None:
                nonlocal calls
                if calls >= 1:
                    raise EvidenceError("Smoke answer exceeded its one-request limit.")
                calls += 1

            evidence = {hit.passage.passage_id: hit.passage for hit in hits}
            answer = ResearchModel(model, reserve).generate(
                SmokeAnswer,
                f"{question} Extract the integer retry_limit and provide a citation. "
                "Quote the exact supporting sentence.\n"
                f"UNTRUSTED EVIDENCE:\n{evidence_context(evidence)}",
            )
            validate_claims(
                [Claim(text=f"Retry limit: {answer.retry_limit}", citations=answer.citations)],
                evidence,
            )
            if answer.retry_limit != EXPECTED_RETRY_LIMIT or not any(
                evidence[c.passage_id].source_id == relevant.source_id
                and EXPECTED_SENTENCE in c.quote
                for c in answer.citations
            ):
                raise EvidenceError("Smoke answer did not cite the expected value of 17.")
            checks[stage] = "passed"
            if web:
                stage = "tavily"
                progress("Smoke: one Tavily search (no result pages fetched)")
                if not discover("Elasticsearch semantic_text official documentation", services):
                    raise EvidenceError("Tavily smoke search returned no usable URLs.")
                checks[stage] = "passed"
        result["status"] = "passed"
    except KeyboardInterrupt:
        checks[stage] = "interrupted"
        result.update(status="interrupted", error="Smoke test interrupted; rerun to retry.")
    except Exception as exc:
        checks[stage] = "failed"
        result.update(
            status="failed",
            error=f"Smoke check '{stage}' failed. Check service access "
            "and model compatibility; no raw service response is displayed.",
        )
        if isinstance(exc, ModelProbeError) and exc.code in PROBE_ERRORS:
            result.update(error_code=exc.code, error=PROBE_ERRORS[exc.code])
        elif isinstance(exc, JevError) and exc.code in JEV_ERRORS:
            result.update(
                error_code=f"jev_{exc.code}",
                error=f"{JEV_ERRORS[exc.code]} No fallback was applied.",
            )
    return result
