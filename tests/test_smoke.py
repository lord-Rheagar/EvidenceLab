import json
from contextlib import contextmanager

import pytest
from test_research import LocalIndex, ScriptedModel

from evidencelab import providers, smoke
from evidencelab.cli import app
from evidencelab.config import AppConfig, ConfigError, ModelProfile, Provider, Services, save_config
from evidencelab.research_model import ResearchModel
from evidencelab.research_records import Citation
from evidencelab.storage import EvidenceStore


@pytest.fixture
def configured(monkeypatch, isolated_settings):
    monkeypatch.setenv("ELASTICSEARCH_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-model-key")
    services = Services(elasticsearch_url="https://elastic.example", inference_id="embedding-test")
    profile = ModelProfile(provider=Provider.OPENAI, model="test-model")
    save_config(
        isolated_settings,
        AppConfig(default_profile="test", profiles={"test": profile}, services=services),
    )
    store = EvidenceStore()
    calls = []
    model = ScriptedModel()
    monkeypatch.setattr(smoke, "build_model", lambda *args: model)
    monkeypatch.setattr(smoke, "check_model", lambda *args: calls.append("probe"))

    @contextmanager
    def open_index(settings):
        yield LocalIndex(store)

    monkeypatch.setattr(smoke, "open_index", open_index)

    def generate(self, schema, prompt):
        self.reserve_call()
        calls.append("answer")
        evidence = json.loads(prompt.split("UNTRUSTED EVIDENCE:\n")[-1])
        p = next(p for p in evidence if "17" in p["text"])
        return schema(
            retry_limit=17,
            citations=[Citation(passage_id=p["passage_id"], quote=p["text"])],
        )

    monkeypatch.setattr(ResearchModel, "generate", generate)
    yield store, profile, services, calls
    store.close()


def test_smoke_checks_and_retained_synthetic_evidence(configured):
    store, profile, services, calls = configured
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["status"] == "passed"
    assert result["checks"] == {
        "model_probe": "passed",
        "index": "passed",
        "retrieval": "passed",
        "cited_answer": "passed",
        "tavily": "skipped",
        "jev": "skipped",
    }
    assert calls == ["probe", "answer"]
    assert store.get_run(result["run_id"]).purpose == "smoke"
    assert len(store.passages(result["run_id"])) == 2


def test_smoke_failure_never_reports_unrun_checks_as_passed(configured, monkeypatch):
    store, profile, services, _ = configured

    def fail(*args):
        raise ConfigError("PRIVATE_TOKEN")

    monkeypatch.setattr(smoke, "check_model", fail)
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["status"] == "failed"
    assert result["checks"]["model_probe"] == "failed"
    assert result["checks"]["index"] == "not_run"
    assert "PRIVATE_TOKEN" not in json.dumps(result)


@pytest.mark.parametrize("code", ["timeout", "rate_limit", "missing_tool_call"])
def test_smoke_preserves_allowlisted_probe_diagnostic(configured, monkeypatch, runner, code):
    def fail(*args):
        error = providers.ModelProbeError(code)
        error.args = ("PRIVATE_PROVIDER_RESPONSE",)
        raise error

    monkeypatch.setattr(smoke, "check_model", fail)
    result = runner.invoke(app, ["smoke-test", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error_code"] == code
    assert data["error"] == providers.PROBE_ERRORS[code]
    assert data["checks"]["index"] == "not_run"
    assert "PRIVATE_PROVIDER_RESPONSE" not in result.output


def test_smoke_preflight_blocks_all_network_if_required_keys_missing(configured, monkeypatch):
    store, profile, services, calls = configured
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        smoke.smoke_test(store, "test", profile, services, True, lambda msg: None)
    assert not calls and not store.runs()
    monkeypatch.setattr(smoke, "build_model", providers.build_model)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert not calls and not store.runs()


def test_smoke_citation_failure_and_cli_json(configured, monkeypatch, runner):
    def invalid(self, schema, prompt):
        return schema(
            retry_limit=17, citations=[Citation(passage_id="0" * 64, quote="Invented quote")]
        )

    monkeypatch.setattr(ResearchModel, "generate", invalid)
    result = runner.invoke(app, ["smoke-test", "--json"])
    assert result.exit_code == 1, result.output
    data = json.loads(result.stdout)
    assert data["checks"]["cited_answer"] == "failed"
    assert data["run_id"]
    assert "API charges" in result.stderr


def test_smoke_web_search_is_opt_in(configured, monkeypatch, runner):
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    queries = []

    def search(question, settings):
        queries.append(question)
        return ["https://elastic.co/"]

    monkeypatch.setattr(smoke, "discover", search)
    result = runner.invoke(app, ["smoke-test", "--web", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["checks"]["tavily"] == "passed"
    assert len(queries) == 1


@pytest.mark.parametrize("flags", [[], ["--no-web"]])
def test_smoke_skips_tavily_without_web_flag(configured, monkeypatch, runner, flags):
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")

    def unexpected(*args):
        pytest.fail("Tavily must not be called without --web")

    monkeypatch.setattr(smoke, "discover", unexpected)
    result = runner.invoke(app, ["smoke-test", *flags, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["checks"]["tavily"] == "skipped"


def test_smoke_web_cli_preflights_key(configured, runner):
    store, _, _, calls = configured
    result = runner.invoke(app, ["smoke-test", "--web", "--json"])
    assert result.exit_code == 2, result.output
    assert "TAVILY_API_KEY" in result.stdout
    assert not calls and not store.runs()


@pytest.mark.parametrize("service_error", [False, True])
def test_smoke_web_failure_exits_nonzero(configured, monkeypatch, runner, service_error):
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    queries = []

    def search(question, settings):
        queries.append(question)
        if service_error:
            raise RuntimeError("PRIVATE_TAVILY_RESPONSE")
        return []

    monkeypatch.setattr(smoke, "discover", search)
    result = runner.invoke(app, ["smoke-test", "--web", "--json"])
    assert result.exit_code == 1, result.output
    data = json.loads(result.stdout)
    assert data["status"] == "failed"
    assert data["checks"]["tavily"] == "failed"
    assert data["checks"]["cited_answer"] == "passed"
    assert len(queries) == 1
    assert "PRIVATE_TAVILY_RESPONSE" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {"retry_limit": 17.5},
        {"retry_limit": 17.0},
        {"retry_limit": "17"},
        {"retry_limit": True},
        {"retry_limit": 99},
        {"text": "The smoke_retry_limit is 17.5 attempts."},
        {"text": "The limit is not 17; it is 99."},
        {"retry_limit": 17, "text": "The limit is not 17; it is 99."},
    ],
)
def test_smoke_rejects_wrong_or_ambiguous_value_with_valid_quote(configured, monkeypatch, payload):
    store, profile, services, _ = configured

    def invalid(self, schema, prompt):
        self.reserve_call()
        evidence = json.loads(prompt.split("UNTRUSTED EVIDENCE:\n")[-1])
        p = next(p for p in evidence if "17" in p["text"])
        return schema.model_validate(
            {**payload, "citations": [{"passage_id": p["passage_id"], "quote": p["text"]}]}
        )

    monkeypatch.setattr(ResearchModel, "generate", invalid)
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["status"] == "failed"
    assert result["checks"]["cited_answer"] == "failed"


def test_smoke_requires_supporting_sentence_not_just_valid_quote(configured, monkeypatch):
    store, profile, services, _ = configured

    def invalid(self, schema, prompt):
        evidence = json.loads(prompt.split("UNTRUSTED EVIDENCE:\n")[-1])
        p = next(p for p in evidence if "17" in p["text"])
        return schema(
            retry_limit=17,
            citations=[
                Citation(passage_id=p["passage_id"], quote="Synthetic EvidenceLab test fixture.")
            ],
        )

    monkeypatch.setattr(ResearchModel, "generate", invalid)
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["checks"]["cited_answer"] == "failed"


def test_smoke_answer_request_budget(configured, monkeypatch):
    store, profile, services, _ = configured

    def too_many(self, *args):
        self.reserve_call()
        self.reserve_call()
        raise AssertionError("Unreachable")

    monkeypatch.setattr(ResearchModel, "generate", too_many)
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["checks"]["cited_answer"] == "failed"


@pytest.mark.parametrize("relevant_first", [True, False])
def test_smoke_jev_selected_profile_checks_expected_order(configured, monkeypatch, relevant_first):
    from evidencelab import reranking
    from evidencelab.config import Reranker
    from evidencelab.reranking import Relevance, RerankAudit, apply_ranking, ranking_key

    store, profile, services, _ = configured
    profile.reranker = Reranker.JEV
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def rank(question, hits, services, *, model):
        ratings = [
            Relevance(
                passage_id=h.passage.passage_id,
                score=3 if ("17" in h.passage.text) == relevant_first else 0,
                confidence=1,
            )
            for h in hits
        ]
        audit = RerankAudit(
            cache_key=ranking_key(question, hits, model), model="jev-test", ratings=ratings
        )
        return apply_ranking(hits, audit), audit

    monkeypatch.setattr(reranking, "rerank", rank)
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["checks"]["jev"] == ("passed" if relevant_first else "failed")
    assert result["checks"]["cited_answer"] == ("passed" if relevant_first else "not_run")
    if not relevant_first:
        assert result["error_code"] == "jev_wrong_order"


@pytest.mark.parametrize("code", ["timeout", "rate_limit", "invalid_response", "score_mismatch"])
def test_smoke_preserves_safe_jev_diagnostic(configured, monkeypatch, code):
    from evidencelab import reranking
    from evidencelab.config import Reranker

    store, profile, services, calls = configured
    profile.reranker = Reranker.JEV
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def fail(*args, **kwargs):
        error = reranking.JevError(code)
        error.args = ("PRIVATE_PROVIDER_RESPONSE",)
        raise error

    monkeypatch.setattr(reranking, "rerank", fail)
    result = smoke.smoke_test(store, "test", profile, services, False, lambda msg: None)
    assert result["checks"]["jev"] == "failed"
    assert result["checks"]["cited_answer"] == "not_run"
    assert result["error_code"] == f"jev_{code}"
    assert "PRIVATE" not in json.dumps(result)
    assert calls == ["probe"]


@pytest.mark.parametrize("value", [17, 17.5, 99])
def test_smoke_installed_model_harness_and_tavily_sdk(monkeypatch, value):
    import httpx
    import requests

    from evidencelab.storage import open_store

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-model-key")
    monkeypatch.setenv("ELASTICSEARCH_API_KEY", "offline-elastic-key")
    monkeypatch.setenv("TAVILY_API_KEY", "offline-search-key")
    model_calls = []
    web_calls = []

    def model_send(self, request, *args, **kwargs):
        body = json.loads(request.content)
        model_calls.append(body)
        names = [tool["function"]["name"] for tool in body["tools"]]
        assert len(names) == 1
        schema = names[0]
        if schema == "CapabilityProbe":
            arguments = {"status": "ok", "value": 7}
        else:
            assert schema == "SmokeAnswer"
            prompt = body["messages"][-1]["content"]
            evidence = json.loads(prompt.split("UNTRUSTED EVIDENCE:\n")[-1])
            p = next(p for p in evidence if "17" in p["text"])
            arguments = {
                "retry_limit": value,
                "citations": [{"passage_id": p["passage_id"], "quote": p["text"]}],
            }
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "offline-response",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "offline-call",
                                    "type": "function",
                                    "function": {
                                        "name": schema,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    def web_send(self, request, **kwargs):
        assert request.url == "https://api.tavily.com/search"
        assert kwargs["timeout"] == 30
        assert request.headers["Authorization"] == "Bearer offline-search-key"
        web_calls.append(json.loads(request.body))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"results": [{"url": "https://www.elastic.co/docs/"}]}'
        return response

    monkeypatch.setattr(httpx.Client, "send", model_send)
    monkeypatch.setattr(requests.Session, "send", web_send)
    with open_store() as store:

        @contextmanager
        def open_index(settings):
            yield LocalIndex(store)

        monkeypatch.setattr(smoke, "open_index", open_index)
        result = smoke.smoke_test(
            store,
            "offline",
            ModelProfile(provider=Provider.OPENROUTER, model="test-model"),
            Services(elasticsearch_url="https://elastic.example", inference_id="embedding-test"),
            True,
            lambda msg: None,
        )
    assert len(model_calls) == 2, result
    assert result["status"] == ("passed" if value == 17 else "failed")
    assert result["checks"]["cited_answer"] == ("passed" if value == 17 else "failed")
    assert result["checks"]["tavily"] == ("passed" if value == 17 else "not_run")
    assert len(web_calls) == (1 if value == 17 else 0)
    if web_calls:
        assert web_calls[0]["search_depth"] == "basic"
        assert web_calls[0]["max_results"] == 5
        assert web_calls[0]["include_answer"] is False
        assert web_calls[0]["include_raw_content"] is False
        assert web_calls[0]["auto_parameters"] is False
