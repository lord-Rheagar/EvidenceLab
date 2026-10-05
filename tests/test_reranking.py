import json
import logging

import httpx2
import pytest
import typesafe_sdk

from evidencelab.config import ConfigError, Services
from evidencelab.documents import extract_source
from evidencelab.evidence import EvidenceError, SearchHit, chunk_source
from evidencelab.reranking import RUBRIC, JevError, rerank


@pytest.fixture
def hits():
    return [
        SearchHit(
            passage=chunk_source(
                extract_source("a" * 32, f"smoke://fixture/{n}", text.encode(), "text", "Fixture")
            )[0],
            score=1 / (60 + n),
            lexical_rank=n + 1,
        )
        for n, text in enumerate(
            [
                "Unrelated document mentioning green and red interface colors.",
                "Keyword search supports exact identifiers and error codes.",
            ]
        )
    ]


def score_response(values):
    return {
        "model": "jev-test-version",
        "usage": {"input_tokens": 50, "output_tokens": 5},
        "answers": {
            f"p{n}": {
                "type": "score",
                "score": float(value),
                "confidence": 1.0,
                "legend": {str(i): text for i, text in enumerate(RUBRIC)},
                "probabilities": {str(i): float(i == value) for i in range(4)},
            }
            for n, value in enumerate(values)
        },
    }


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://untrusted.example")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "wrong-model")
    actual = typesafe_sdk.TypeSafeClient
    requests = []
    response = score_response([0, 3])
    status = [200]

    def handle(request):
        requests.append(request)
        # Raw content lets malformed values reach SDK validation, including NaN.
        return httpx2.Response(
            status[0],
            content=json.dumps(response).encode(),
            headers={"content-type": "application/json"},
        )

    def client(**kwargs):
        assert kwargs["retry"].max_retries == 0
        return actual(**kwargs, transport=httpx2.MockTransport(handle))

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", client)
    return requests, response, status


def test_real_sdk_scoring_preserves_ids_and_ignores_ambient_endpoints(hits, transport):
    requests, _, _ = transport
    before = [h.model_dump() for h in hits]
    ordered, audit = rerank("Find exact identifiers", hits, Services(), model="jev-pinned")
    assert ordered[0].passage == hits[1].passage
    assert ordered[0].rerank_score == 3
    assert ordered[0].score == hits[1].score
    assert [h.model_dump() for h in hits] == before
    assert audit.model == "jev-test-version"
    assert audit.excerpt_chars == 4000
    assert str(requests[0].url) == "https://api.typesafe.ai/v1/systemone"
    body = json.loads(requests[0].content)
    assert body["model"] == "jev-pinned"
    assert set(body["questions"]) == {"p0", "p1"}
    assert body["state"] == {"research_question": "Find exact identifiers"}
    instructions = body["questions"]["p0"]["instructions"]
    assert instructions["candidate_passage"] == hits[0].passage.text
    assert hits[1].passage.text not in json.dumps(instructions)
    assert hits[0].passage.location not in json.dumps(body)


def test_reranker_sees_the_same_4000_character_window_as_the_reviewer(hits, transport):
    from evidencelab.reranking import ranking_key

    text = "x" * 2045 + "Asynchronously run function in a separate thread." + "z" * 2200
    hits[0] = hits[0].model_copy(
        update={"passage": hits[0].passage.model_copy(update={"text": text})}
    )
    before = ranking_key("Question", hits, "jev-pinned")
    rerank("Question", hits, Services(), model="jev-pinned")
    body = json.loads(transport[0][0].content)
    passage = body["questions"]["p0"]["instructions"]["candidate_passage"]
    assert "Asynchronously run function" in passage
    assert text[:4000] == passage
    assert text != passage
    changed = [
        hits[0].model_copy(
            update={
                "passage": hits[0].passage.model_copy(
                    update={"text": text[:2045] + "DIFFERENT LATE CONTENT" + text[2092:]}
                )
            }
        ),
        hits[1],
    ]
    assert ranking_key("Question", changed, "jev-pinned") != before


def test_ranking_cache_does_not_depend_on_hit_order(hits):
    from evidencelab.reranking import ranking_key

    assert ranking_key("query", hits, "jev-pinned") == ranking_key(
        "query", hits[::-1], "jev-pinned"
    )


@pytest.mark.parametrize(
    ("score", "valid"),
    [
        (0.92, True),
        (0.88, True),
        (0.90, True),
        (0.919, True),
        (0.92000001, False),
        (0.87999999, False),
    ],
)
def test_score_tolerance_includes_float_boundary_but_not_larger_errors(
    hits, transport, score, valid
):
    requests, response, _ = transport
    # Live Jev response: weighted score 0.90, reported score 0.92.
    response["answers"]["p0"].update(
        score=score,
        probabilities={"0": 0.34, "1": 0.43, "2": 0.22, "3": 0.01},
    )
    if valid:
        ordered, audit = rerank("query", hits, Services())
        assert audit.ratings[0].score == score
        assert ordered[1].rerank_score == score
        assert ordered[1].passage == hits[0].passage
    else:
        with pytest.raises(JevError) as caught:
            rerank("query", hits, Services())
        assert caught.value.code == "score_mismatch"
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("total", "valid"),
    [(0.98, True), (1.02, True), (0.97999999, False), (1.02000001, False)],
)
def test_probability_sum_tolerance_handles_float_boundary(hits, transport, total, valid):
    requests, response, _ = transport
    response["answers"]["p0"].update(
        score=0.90,
        probabilities={"0": total - 0.66, "1": 0.43, "2": 0.22, "3": 0.01},
    )
    if valid:
        _, audit = rerank("query", hits, Services())
        assert audit.ratings[0].score == 0.90
    else:
        with pytest.raises(JevError) as caught:
            rerank("query", hits, Services())
        assert caught.value.code == "invalid_probabilities"
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("bad", "code"),
    [
        ("missing", "candidate_mismatch"),
        ("extra", "candidate_mismatch"),
        ("nan", "score_mismatch"),
        ("range", "invalid_response"),
        ("rubric", "rubric_mismatch"),
        ("probabilities", "invalid_probabilities"),
        ("score", "score_mismatch"),
    ],
)
def test_invalid_scores_fail_closed(hits, transport, bad, code):
    requests, response, _ = transport
    if bad == "missing":
        response["answers"].pop("p0")
    elif bad == "extra":
        response["answers"]["invented"] = response["answers"]["p0"]
    elif bad == "nan":
        response["answers"]["p0"]["score"] = float("nan")
    elif bad == "range":
        response["answers"]["p0"]["confidence"] = 2
    elif bad == "rubric":
        response["answers"]["p0"]["legend"]["0"] = "Different criteria"
    elif bad == "score":
        response["answers"]["p0"]["score"] = 1.5
    else:
        response["answers"]["p0"]["probabilities"]["0"] = 0.5
    with pytest.raises(JevError, match="No fallback") as error:
        rerank("query", hits, Services())
    assert error.value.code == code
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("http_status", "code"),
    [
        (401, "authentication"),
        (403, "access_denied"),
        (402, "quota"),
        (429, "rate_limit"),
        (400, "request_rejected"),
        (404, "request_rejected"),
        (422, "request_rejected"),
        (500, "provider_unavailable"),
        (503, "provider_unavailable"),
        (408, "timeout"),
        (504, "timeout"),
    ],
)
def test_service_error_has_no_hidden_retry_or_raw_error(hits, transport, http_status, code):
    requests, response, status = transport
    status[0] = http_status
    response.clear()
    response.update(error={"message": "PRIVATE_DOCUMENT_AND_API_SECRET"})
    with pytest.raises(JevError) as error:
        rerank("query", hits, Services())
    assert error.value.code == code
    assert "PRIVATE" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("timeout", [True, False])
def test_sdk_transport_failures_are_classified_and_not_retried(hits, monkeypatch, timeout):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    actual = typesafe_sdk.TypeSafeClient
    requests = []
    logger = logging.getLogger("typesafe_sdk")
    was_disabled = logger.disabled

    def handle(request):
        requests.append(request)
        error = httpx2.ReadTimeout if timeout else httpx2.ConnectError
        raise error("PRIVATE_PROVIDER_RESPONSE", request=request)

    def client(**kwargs):
        assert kwargs["timeout"] == 30
        assert kwargs["retry"].max_retries == 0
        return actual(**kwargs, transport=httpx2.MockTransport(handle))

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", client)
    with pytest.raises(JevError) as error:
        rerank("query", hits, Services())
    assert error.value.code == ("timeout" if timeout else "connection")
    assert "PRIVATE" not in str(error.value)
    assert len(requests) == 1
    assert logger.disabled == was_disabled


def test_sdk_invalid_response_is_classified(hits, transport):
    _, response, _ = transport
    response["answers"]["p0"].pop("confidence")
    with pytest.raises(JevError) as error:
        rerank("query", hits, Services())
    assert error.value.code == "invalid_response"


def test_ties_preserve_hybrid_order_and_logging_is_restored(hits, transport):
    _, response, _ = transport
    response.update(score_response([3, 3]))
    logger = logging.getLogger("typesafe_sdk")
    disabled = logger.disabled
    ordered, _ = rerank("query", hits, Services())
    assert [h.passage for h in ordered] == [h.passage for h in hits]
    assert logger.disabled == disabled


def test_missing_credential_and_duplicate_candidates_do_not_call_sdk(hits):
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        rerank("query", hits, Services())
    with pytest.raises(EvidenceError, match="Duplicate"):
        rerank("query", [hits[0], hits[0]], Services())


def test_failed_score_diagnostic_retains_only_numbers(hits, transport):
    _, response, _ = transport
    response["answers"]["p0"]["score"] = 1.5
    events = []
    with pytest.raises(JevError, match="probability distribution"):
        rerank("PRIVATE_QUESTION", hits, Services(), diagnostic=events.append)
    assert len(events) == 1
    event = events[0]
    assert event.code == "score_mismatch"
    assert event.score == 1.5 and event.weighted_score == 0
    assert event.probabilities == [1, 0, 0, 0]
    assert event.candidate_index == 0 and event.elapsed_seconds >= 0
    serialized = event.model_dump_json()
    for forbidden in ["PRIVATE", "test-key", hits[0].passage.text, RUBRIC[0]]:
        assert forbidden not in serialized


def test_independent_rounding_reproduces_old_false_rejection(hits, transport):
    import math

    exact = [0.0951, 0.10497, 0.30497, 0.49496]
    assert math.isclose(sum(exact), 1)
    score = round(sum(k * p for k, p in enumerate(exact)), 2)
    probabilities = {str(k): round(p, 2) for k, p in enumerate(exact)}
    weighted = sum(int(k) * p for k, p in probabilities.items())
    assert score == 2.20 and math.isclose(weighted, 2.17)
    assert not math.isclose(score, weighted, rel_tol=0, abs_tol=0.02 + 1e-12)
    transport[1]["answers"]["p0"].update(score=score, probabilities=probabilities)
    events = []
    _, audit = rerank("query", hits, Services(), diagnostic=events.append)
    assert audit.ratings[0].score == score  # Never rewrite the provider's score.
    assert events[0].rounding_matches == 1


@pytest.mark.parametrize("score", [2.21, 2.20000001, -1, float("nan"), float("inf")])
def test_rounding_allowance_rejects_infeasible_or_high_precision_scores(score):
    from evidencelab.reranking import rounded_score_matches

    assert not rounded_score_matches(score, {0: 0.10, 1: 0.10, 2: 0.30, 3: 0.49})


def test_rounding_feasibility_handles_edges_and_random_normalized_distributions():
    import random

    from evidencelab.reranking import rounded_score_matches

    rng = random.Random(47)
    for _ in range(1000):
        raw = [rng.random() for _ in range(4)]
        exact = [p / sum(raw) for p in raw]
        score = round(sum(k * p for k, p in enumerate(exact)), 2)
        probabilities = {k: round(p, 2) for k, p in enumerate(exact)}
        assert rounded_score_matches(score, probabilities)
    assert rounded_score_matches(0, {0: 1, 1: 0, 2: 0, 3: 0})
    assert rounded_score_matches(3, {0: 0, 1: 0, 2: 0, 3: 1})
    assert not rounded_score_matches(0.03, {0: 1, 1: 0, 2: 0, 3: 0})
    assert not rounded_score_matches(0, {0: 0.5, 1: 0, 2: 0, 3: 0})


def test_jev_http_diagnostics_keep_status_not_body(hits, transport):
    _, response, status = transport
    status[0] = 429
    response.clear()
    response["error"] = {"message": "PRIVATE_SECRET"}
    events = []
    with pytest.raises(JevError):
        rerank("query", hits, Services(), diagnostic=events.append)
    assert events[0].code == "rate_limit" and events[0].http_status == 429
    assert "PRIVATE" not in events[0].model_dump_json()


def test_context_rejection_reports_allowlisted_reason_without_response_text(hits, transport):
    requests, response, status = transport
    status[0] = 400
    response.clear()
    response.update(detail={"error_type": "max_tokens_exceeded", "input": "PRIVATE_SECRET"})
    events = []
    with pytest.raises(JevError, match="context limit") as error:
        rerank("query", hits, Services(), diagnostic=events.append)
    assert error.value.code == "context_limit"
    assert len(requests) == 1
    assert events[0].http_status == 400 and events[0].code == "context_limit"
    assert events[0].candidate_count == 2
    assert events[0].request_bytes > events[0].state_bytes > 0
    assert "PRIVATE" not in events[0].model_dump_json()


def test_multilingual_batches_preserve_complete_windows_and_cache_boundaries(hits):
    from evidencelab.evidence import digest
    from evidencelab.reranking import candidate_batches, payload_fits, request_payload

    candidates = [
        hits[0].model_copy(
            update={
                "passage": hits[0].passage.model_copy(
                    update={
                        "passage_id": digest(str(n)),
                        "text": "\U0001f600" * 4000 + str(n),
                    }
                )
            }
        )
        for n in range(12)
    ]
    batches = candidate_batches("Question", candidates, "jev-latest")
    assert 1 < len(batches) < 12
    assert [[h.passage.passage_id for h in b] for b in batches] == [
        [h.passage.passage_id for h in b]
        for b in candidate_batches("Question", candidates[::-1], "jev-latest")
    ]
    assert {h.passage.passage_id for b in batches for h in b} == {
        h.passage.passage_id for h in candidates
    }
    for batch in batches:
        payload = request_payload("Question", batch, "jev-latest")
        assert payload_fits(payload)
        assert payload["state"] == {"research_question": "Question"}
        for n, hit in enumerate(batch):
            assert (
                payload["questions"][f"p{n}"]["instructions"]["candidate_passage"]
                == (hit.passage.text[:4000])
            )


def test_oversized_single_context_is_rejected_before_network(hits, transport):
    from evidencelab.reranking import candidate_batches

    question = "\U0001f600" * 8000
    with pytest.raises(JevError, match="context limit"):
        candidate_batches(question, hits, "jev-latest")
    with pytest.raises(JevError, match="context limit"):
        rerank(question, hits, Services())
    assert transport[0] == []


def test_jev_model_selection_survives_changing_generator(runner, isolated_settings):
    from evidencelab.cli import app
    from evidencelab.config import load_config

    result = runner.invoke(
        app,
        [
            "configure",
            "--non-interactive",
            "--provider",
            "openai",
            "--model",
            "test",
            "--reranker",
            "jev",
            "--jev-model",
            "jev-pinned",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["configure", "--non-interactive", "--provider", "anthropic", "--model", "other-model"]
    )
    assert result.exit_code == 0, result.output
    profile = load_config(isolated_settings).profiles["default"]
    assert profile.jev_model == "jev-pinned" and profile.reranker == "jev"
