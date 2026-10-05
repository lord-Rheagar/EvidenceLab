import json
from contextlib import contextmanager

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from evidencelab import research, retrieval
from evidencelab.cli import app
from evidencelab.config import (
    AppConfig,
    Depth,
    ModelProfile,
    Provider,
    Reranker,
    Services,
    save_config,
)
from evidencelab.documents import extract_source
from evidencelab.evidence import EvidenceError, SearchHit
from evidencelab.reports import export_report, render_report
from evidencelab.research import ResearchPipeline, create_job
from evidencelab.research_records import LIMITS
from evidencelab.storage import EvidenceStore


class ScriptedModel(BaseChatModel):
    model_name: str = "research-test-model"
    seen_tools: list[list[str]] = Field(default_factory=list)
    seen_prompts: list[str] = Field(default_factory=list)
    bad_citation: bool = False
    invalid_review_schema: bool = False
    invalid_report_schema: bool = False
    fail_schema: str = ""
    repair_schema: str = ""
    seen_feedback: list[str] = Field(default_factory=list)
    gaps: bool = False
    angle_count: int = 2
    prohibited_tool: bool = False

    @property
    def _llm_type(self):
        return "scripted-research"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tools = [t["function"]["name"] for t in kwargs["tools"]]
        self.seen_tools.append(tools)
        prompt = next(message.content for message in reversed(messages) if message.type == "human")
        if messages[-1].type == "tool":
            self.seen_feedback.append(messages[-1].content)
        self.seen_prompts.append(prompt)
        schema = tools[0]
        if self.prohibited_tool:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "read_file",
                                    "args": {"file_path": "/private.txt"},
                                    "id": "bad",
                                }
                            ],
                        )
                    )
                ]
            )
        if self.fail_schema == schema:
            raise RuntimeError("private provider error with API_SECRET")
        if schema == "ResearchPlan":
            data = {
                "angles": [f"Technical angle {n}" for n in range(self.angle_count)],
                "queries": []
                if "Web search enabled: False" in prompt
                else ["official retrieval docs"],
            }
        else:
            evidence = json.loads(prompt.split("UNTRUSTED EVIDENCE:\n")[-1])[0]
            claim = {
                "text": "Identifiers can be matched with keyword search.",
                "citations": [
                    {
                        "excerpt_id": "PRIVATE_INVENTED_REFERENCE"
                        if self.bad_citation
                        else evidence["excerpts"][0]["excerpt_id"],
                    }
                ],
            }
            if schema == "EvidenceReview":
                if self.invalid_review_schema:
                    claim["citations"][0]["excerpt_id"] = "PRIVATE_INVALID_PASSAGE_ID"
                data = {
                    "findings": [claim],
                    "gaps": ["Performance measurements are missing."],
                    "follow_up_queries": ["official benchmarks"] if self.gaps else [],
                }
            else:
                if self.invalid_report_schema:
                    claim["citations"][0]["excerpt_id"] = "PRIVATE_INVALID_PASSAGE_ID"
                data = {
                    "title": "Technical retrieval research",
                    "summary": [claim],
                    "findings": [claim],
                    "tradeoffs": [],
                    "limitations": ["Limited to the provided evidence."],
                }
        if (
            self.repair_schema == schema
            and tools in self.seen_tools
            and self.seen_tools.count(tools) == 1
        ):
            if schema == "ResearchPlan":
                data["angles"] = ["Too many angles"] * 6
            elif schema == "EvidenceReview":
                data["gaps"] = ["PRIVATE_GAP_VALUE"] * 9
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="", tool_calls=[{"name": schema, "args": data, "id": "result"}]
                    )
                )
            ]
        )


class LocalIndex:
    def __init__(self, store):
        self.store = store
        self.index_calls = 0
        self.search_calls = 0
        self.tamper = False

    def index_run(self, store, run_id):
        self.index_calls += 1
        run = store.get_run(run_id)
        run.status = "indexed"
        store.save_run(run)

    def search(self, run, question, limit):
        self.search_calls += 1
        passages = self.store.passages(run.run_id)[:limit]
        if self.tamper:
            passages = [p.model_copy(update={"text": "Malicious replacement"}) for p in passages]
        return [SearchHit(passage=p, score=1) for p in passages]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("ELASTICSEARCH_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-search-key")
    services = Services(elasticsearch_url="https://elastic.example", inference_id="embeddings")
    profile = ModelProfile(provider=Provider.OPENAI, model="test-model", depth=Depth.QUICK)
    file = tmp_path / "notes.txt"
    file.write_text(
        "Keyword search supports exact identifiers. Semantic search finds related ideas."
    )
    store = EvidenceStore()
    model = ScriptedModel()
    monkeypatch.setattr(research, "build_model", lambda *args: model)
    yield store, services, profile, file, model
    store.close()


def make_job(setup, **kwargs):
    store, services, profile, file, _ = setup
    return create_job(
        store,
        "Which retrieval approach fits?",
        "chosen",
        profile,
        services,
        [file],
        [],
        kwargs.get("web", False),
    )


@pytest.mark.parametrize("failure", ["service", "budget"])
def test_jev_batches_checkpoint_each_request_and_respect_budget(setup, monkeypatch, failure):
    from evidencelab import reranking
    from evidencelab.evidence import chunk_source
    from evidencelab.reranking import Relevance, RerankAudit, apply_ranking, ranking_key
    from evidencelab.research_records import ResearchPlan

    store, services, profile, _, _ = setup
    profile.reranker = Reranker.JEV
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    job = make_job(setup)
    passages = [
        chunk_source(
            extract_source(
                job.run_id, f"smoke://batch/{n}", ("x" * 3000 + str(n)).encode(), "text", "Fixture"
            )
        )[0]
        for n in range(2)
    ]
    hits = [SearchHit(passage=p, score=1) for p in passages]
    monkeypatch.setattr(reranking, "REQUEST_BYTES", 4500)
    batches = reranking.candidate_batches("Query", hits, profile.jev_model)
    assert len(batches) == 2
    job.plan = ResearchPlan(angles=["Query", "No matches"], queries=[])
    job.stage = "review"
    job.rerank_calls = 5 if failure == "budget" else 0
    store.save_job(job)
    index = LocalIndex(store)
    monkeypatch.setattr(index, "search", lambda run, query, limit: hits if query == "Query" else [])
    calls = []

    def fake_rerank(question, batch, services, *, model, diagnostic=None):
        calls.append(batch[0].passage.passage_id)
        assert store.get_job(job.run_id).rerank_calls == job.rerank_calls
        if failure == "service" and len(calls) == 2:
            raise EvidenceError("Jev unavailable")
        audit = RerankAudit(
            cache_key=ranking_key(question, batch, model),
            model=model,
            ratings=[
                Relevance(passage_id=h.passage.passage_id, score=len(calls), confidence=1)
                for h in batch
            ],
        )
        return apply_ranking(batch, audit), audit

    monkeypatch.setattr(research, "rerank", fake_rerank)
    pipeline = ResearchPipeline(store, job, services, index, lambda message: None)
    saved = {p.passage_id: p for p in passages}
    with pytest.raises(EvidenceError):
        pipeline.select_review_evidence(saved)
    assert len(store.get_job(job.run_id).rerankings) == 1
    if failure == "budget":
        assert job.rerank_calls == 6 and len(calls) == 1
        with pytest.raises(EvidenceError, match="budget exhausted"):
            pipeline.select_review_evidence(saved)
        assert len(calls) == 1
    else:
        result = pipeline.select_review_evidence(saved)
        assert len(calls) == 3 and calls[1] == calls[2]
        assert list(result) == [batches[1][0].passage.passage_id, batches[0][0].passage.passage_id]
        assert len(store.get_job(job.run_id).rerankings) == 2
        pipeline.select_review_evidence(saved)
        assert len(calls) == 3


def test_diagnostics_persist_across_failed_and_completed_resumes(setup):
    store, _, _, _, model = setup
    job = make_job(setup)
    model.fail_schema = "ResearchReport"
    with pytest.raises(EvidenceError, match="model request failed"):
        run_job(setup, job)
    saved = store.get_job(job.run_id)
    assert saved.diagnostics[-1].code == "stage_failed"
    assert saved.diagnostics[-2].code == "unknown"
    assert saved.diagnostics[-2].stage == "write"
    assert saved.diagnostics[-2].schema_name == "ResearchReport"
    assert saved.diagnostics[-2].model_calls == 3
    assert "API_SECRET" not in saved.model_dump_json()
    previous = len(saved.diagnostics)
    model.fail_schema = ""
    run_job(setup, saved)
    complete = store.get_job(job.run_id)
    assert complete.status == "complete" and len(complete.diagnostics) > previous
    stages = [e.stage for e in complete.diagnostics if e.operation == "stage" and e.code == "ok"]
    assert stages == ["plan", "gather", "index", "review", "write"]
    assert store.job_summary(job.run_id)["diagnostics"]
    run_job(setup, complete)
    assert store.get_job(job.run_id) == complete


def test_diagnostics_cli_is_local_and_does_not_dump_research(setup, runner):
    store, _, _, _, _ = setup
    job = make_job(setup)
    run_job(setup, job)
    before = store.get_job(job.run_id)
    result = runner.invoke(app, ["diagnostics", job.run_id, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["diagnostics"]
    assert "question" not in payload and "research" not in payload
    assert "Keyword search supports exact identifiers" not in result.output
    assert store.get_job(job.run_id) == before
    result = runner.invoke(app, ["diagnostics", job.run_id])
    assert "pipeline/stage: ok" in result.output


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
def test_report_printing_handles_legacy_windows_stdout(monkeypatch, encoding):
    import io

    from evidencelab import cli

    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding)
    report = "Research: \u4e2d\u6587 and asyncio"
    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", stream)
        cli.echo_report(report)
        stream.flush()
    output = buffer.getvalue().decode(encoding)
    expected = report if encoding == "utf-8" else "Research: \\u4e2d\\u6587 and asyncio"
    assert output.strip() == expected


def test_old_jobs_and_diagnostic_retention(setup):
    from evidencelab.diagnostics import Diagnostic
    from evidencelab.research_records import ResearchJob

    store, services, _, _, _ = setup
    job = make_job(setup)
    restored = ResearchJob.model_validate(job.model_dump(exclude={"diagnostics"}))
    assert restored.diagnostics == []
    job.diagnostics = [Diagnostic(service="model", code="ok") for _ in range(500)]
    pipeline = ResearchPipeline(store, job, services, LocalIndex(store), lambda message: None)
    pipeline.record_diagnostic(Diagnostic(service="model", code="timeout"))
    saved = store.get_job(job.run_id)
    assert len(saved.diagnostics) == 500 and saved.diagnostics[-1].code == "timeout"


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_model_finish_reason_fails_closed_without_leaking_metadata(setup, monkeypatch, reason):
    original = ScriptedModel._generate

    def truncated(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.generations[0].message.response_metadata = {
            "finish_reason": reason,
            "provider": "PRIVATE_KEY",
        }
        return result

    monkeypatch.setattr(ScriptedModel, "_generate", truncated)
    job = make_job(setup)
    with pytest.raises(EvidenceError, match="model response failed"):
        run_job(setup, job)
    event = job.diagnostics[0]
    assert event.finish_reason == reason
    assert event.code == ("output_truncated" if reason == "length" else "content_filter")
    assert "PRIVATE" not in job.model_dump_json()
    assert job.model_calls == 1


def run_job(setup, job, index=None, **kwargs):
    store, services, _, _, _ = setup
    with store.lock(job.run_id):
        return ResearchPipeline(
            store, job, services, index or LocalIndex(store), lambda message: None, **kwargs
        ).run()


def test_real_deepagents_graph_produces_cited_report_and_offline_exports(setup, tmp_path):
    store, _, _, _, model = setup
    job = run_job(setup, make_job(setup))
    assert job.status == "complete"
    assert job.model_calls == 3
    assert job.search_calls == 0
    assert model.seen_tools == [["ResearchPlan"], ["EvidenceReview"], ["ResearchReport"]]
    assert store.get_job(job.run_id) == job
    for prompt in model.seen_prompts[1:]:
        evidence = json.loads(prompt.split("UNTRUSTED EVIDENCE:\n")[-1])
        assert evidence[0]["passage_id"] == "p1"
        assert job.evidence_ids[0] not in prompt
    assert job.review.findings[0].citations[0].passage_id == job.evidence_ids[0]
    assert job.report.findings[0].citations[0].passage_id == job.evidence_ids[0]
    markdown = render_report(store, job)
    assert "[1]" in markdown
    assert "Keyword search supports exact identifiers." in markdown
    assert "Performance measurements are missing." in markdown
    output = tmp_path / "report.md"
    export_report(store, job, output, "markdown")
    assert output.read_text() == markdown
    with pytest.raises(EvidenceError, match="already exists"):
        export_report(store, job, output, "markdown")
    output = tmp_path / "report.json"
    export_report(store, job, output, "json")
    assert json.loads(output.read_text())["evidence"][0]["text"]


def test_failed_stage_resumes_without_repeating_planning_or_indexing(setup):
    store, _, _, _, model = setup
    job = make_job(setup)
    index = LocalIndex(store)
    model.fail_schema = "ResearchReport"
    with pytest.raises(EvidenceError, match="model request failed"):
        run_job(setup, job, index)
    failed = store.get_job(job.run_id)
    assert failed.stage == "write" and failed.status == "failed"
    assert failed.model_calls == 3
    assert "API_SECRET" not in failed.model_dump_json()
    model.fail_schema = ""
    finished = run_job(setup, failed, index)
    assert finished.model_calls == 4
    assert index.index_calls == 1
    assert len([tools for tools in model.seen_tools if tools == ["ResearchPlan"]]) == 1


def test_failed_review_reuses_its_saved_input_without_searching_again(setup):
    store, _, _, _, model = setup
    job = make_job(setup)
    index = LocalIndex(store)
    model.fail_schema = "EvidenceReview"
    with pytest.raises(EvidenceError, match="model request failed"):
        run_job(setup, job, index)
    saved = store.get_job(job.run_id)
    assert saved.review_snapshot_key
    searches = index.search_calls
    index.tamper = True
    model.fail_schema = ""
    completed = run_job(setup, saved, index)
    assert completed.status == "complete"
    assert index.search_calls == searches


def test_review_input_refreshes_when_new_evidence_is_added(setup):
    store, _, _, _, model = setup
    job = make_job(setup)
    index = LocalIndex(store)
    model.fail_schema = "EvidenceReview"
    with pytest.raises(EvidenceError, match="model request failed"):
        run_job(setup, job, index)
    saved = store.get_job(job.run_id)
    previous = saved.review_snapshot_key
    searches = index.search_calls
    store.add_source(
        extract_source(
            job.run_id,
            "https://docs.example/new",
            b"Additional relevant evidence about retrieval.",
            "text",
            "New evidence",
        )
    )
    index.index_run(store, job.run_id)
    model.fail_schema = ""
    completed = run_job(setup, saved, index)
    assert completed.status == "complete"
    assert completed.review_snapshot_key != previous
    assert index.search_calls > searches


def test_fabricated_citations_block_publication_and_can_be_retried(setup):
    store, _, _, _, model = setup
    job = make_job(setup)
    model.bad_citation = True
    with pytest.raises(EvidenceError, match="invalid EvidenceReview structured output"):
        run_job(setup, job)
    failed = store.get_job(job.run_id)
    assert failed.report is None
    with pytest.raises(EvidenceError, match="not complete"):
        render_report(store, failed)
    model.bad_citation = False
    assert run_job(setup, failed).status == "complete"


def test_review_schema_error_is_actionable_redacted_and_resumable(setup):
    store, _, _, _, model = setup
    model.invalid_review_schema = True
    job = make_job(setup)
    with pytest.raises(EvidenceError, match="invalid EvidenceReview structured output"):
        run_job(setup, job)
    failed = store.get_job(job.run_id)
    assert failed.stage == "review" and failed.status == "failed"
    assert failed.model_calls == 3
    assert len(model.seen_feedback) == 1
    assert "after one correction attempt" in failed.error
    assert "findings.0.citations.0.excerpt_id" in failed.error
    assert "select a reference from the supplied evidence" in failed.error
    assert "PRIVATE_INVALID_PASSAGE_ID" not in failed.model_dump_json()
    model.invalid_review_schema = False
    completed = run_job(setup, failed)
    assert completed.status == "complete" and completed.model_calls == 5


def test_review_and_report_prompts_include_exact_citation_and_size_rules(setup):
    model = setup[-1]
    run_job(setup, make_job(setup))
    for prompt in model.seen_prompts[1:]:
        assert "select a supplied excerpt_id" in prompt
        assert "do not write a quote or passage_id" in prompt
        assert "If no supplied excerpt supports a claim, omit the claim" in prompt
        assert "at most 2000 characters" in prompt


def test_invalid_report_reference_is_redacted_and_resumes_from_saved_canonical_review(setup):
    store, _, _, _, model = setup
    model.invalid_report_schema = True
    index = LocalIndex(store)
    job = make_job(setup)
    with pytest.raises(EvidenceError, match="invalid ResearchReport structured output"):
        run_job(setup, job, index)
    failed = store.get_job(job.run_id)
    assert failed.stage == "write" and failed.report is None
    assert "PRIVATE_INVALID_PASSAGE_ID" not in failed.model_dump_json()
    assert failed.review.findings[0].citations[0].passage_id == failed.evidence_ids[0]
    model.invalid_report_schema = False
    completed = run_job(setup, failed, index)
    assert completed.status == "complete" and completed.model_calls == 5
    assert index.index_calls == 1


def test_schema_diagnostics_hide_values_and_unknown_field_names():
    from pydantic import ValidationError

    from evidencelab.research_model import schema_failure_details
    from evidencelab.research_records import EvidenceReview

    with pytest.raises(ValidationError) as caught:
        EvidenceReview.model_validate(
            {
                "findings": [{"text": "PRIVATE_VALUE", "citations": []}],
                "gaps": [],
                "follow_up_queries": [],
                "PRIVATE_FIELD": "PRIVATE_VALUE",
            }
        )
    wrapped = ValueError("PRIVATE_EXCEPTION_TEXT")
    wrapped.__cause__ = caught.value
    details = schema_failure_details(wrapped, EvidenceReview)
    assert "findings.0.citations: too few items" in details
    assert "unknown_field: unexpected field" in details
    assert "PRIVATE" not in details


@pytest.mark.parametrize("unknown_id", [False, True])
def test_citation_error_identifies_position_and_reason_without_quote(setup, unknown_id):
    from evidencelab.research import validate_claims
    from evidencelab.research_records import Citation, Claim

    store, _, _, file, _ = setup
    job = make_job(setup)
    source = extract_source(job.run_id, file.as_uri(), b"Original document content.", "text", "Doc")
    store.add_source(source)
    passage = store.passages(job.run_id)[0]
    claim = Claim(
        text="A claim",
        citations=[
            Citation(
                passage_id="0" * 64 if unknown_id else passage.passage_id,
                quote="PRIVATE_INVENTED_QUOTATION",
            )
        ],
    )
    with pytest.raises(EvidenceError) as caught:
        validate_claims([claim], {passage.passage_id: passage})
    assert "claim 1, citation 1" in str(caught.value)
    assert ("unknown passage ID" if unknown_id else "not an exact substring") in str(caught.value)
    assert "PRIVATE" not in str(caught.value)


def test_model_budget_survives_resume_and_blocks_calls(setup):
    store, _, profile, _, model = setup
    job = make_job(setup)
    job.model_calls = LIMITS[profile.depth][4]
    store.save_job(job)
    with pytest.raises(EvidenceError, match="budget exhausted"):
        run_job(setup, job)
    assert not model.seen_tools
    assert store.get_job(job.run_id).model_calls == 30


@pytest.mark.parametrize("schema", ["ResearchPlan", "EvidenceReview"])
def test_one_schema_correction_is_counted_and_can_complete(setup, schema):
    store, _, _, _, model = setup
    model.repair_schema = schema
    job = make_job(setup)
    notices = []
    pipeline = ResearchPipeline(store, job, setup[1], LocalIndex(store), notices.append)
    with store.lock(job.run_id):
        completed = pipeline.run()
    assert completed.status == "complete" and completed.model_calls == 4
    assert model.seen_tools.count([schema]) == 2
    assert len(model.seen_feedback) == 1
    assert "PRIVATE_GAP_VALUE" not in model.seen_feedback[0]
    assert any("correction" in notice for notice in notices)


def test_schema_correction_cannot_exceed_the_persisted_budget(setup):
    store, _, profile, _, model = setup
    model.repair_schema = "ResearchPlan"
    job = make_job(setup)
    job.model_calls = LIMITS[profile.depth][4] - 1
    with pytest.raises(EvidenceError, match="budget exhausted"):
        run_job(setup, job)
    assert store.get_job(job.run_id).model_calls == LIMITS[profile.depth][4]
    assert model.seen_tools == [["ResearchPlan"]]


@pytest.mark.parametrize(
    "depth,low,high", [(Depth.QUICK, 2, 3), (Depth.BALANCED, 3, 4), (Depth.THOROUGH, 4, 6)]
)
def test_planner_schema_matches_depth_and_no_web_constraints(depth, low, high):
    from pydantic import ValidationError

    from evidencelab.research_records import plan_schema

    schema = plan_schema(depth, web=False)
    assert schema.__name__ == "ResearchPlan"
    schema.model_validate({"angles": ["Angle"] * low, "queries": []})
    for count in [low - 1, high + 1]:
        with pytest.raises(ValidationError):
            schema.model_validate({"angles": ["Angle"] * count, "queries": []})
    with pytest.raises(ValidationError):
        schema.model_validate({"angles": ["Angle"] * low, "queries": ["No web allowed"]})


def test_remote_tampering_blocked_before_evidence_reaches_model(setup):
    store, _, _, _, model = setup
    job = make_job(setup)
    index = LocalIndex(store)
    index.tamper = True
    with pytest.raises(EvidenceError, match="does not match"):
        run_job(setup, job, index)
    assert model.seen_tools == [["ResearchPlan"]]


def test_web_discovery_snapshots_results_and_bounds_gap_passes(setup, monkeypatch):
    store, services, profile, file, model = setup
    profile.depth = Depth.BALANCED
    model.angle_count, model.gaps = 3, True
    job = make_job(setup, web=True)
    queries = []

    def search(query, settings):
        queries.append(query)
        return [f"https://docs.example/{len(queries)}"]

    def read(run_id, url):
        return extract_source(run_id, url, f"Technical source at {url}".encode(), "text", "Docs")

    monkeypatch.setattr(research, "read_url", read)
    result = run_job(setup, job, search_web=search)
    assert result.search_calls == 2
    assert result.gap_passes == 1
    assert len(store.sources(job.run_id)) == 3
    assert queries == ["official retrieval docs", "official benchmarks"]


def test_search_failure_charged_to_budget_and_retry_is_explicit(setup):
    store, _, _, _, _ = setup
    job = make_job(setup, web=True)

    def fail(*args):
        raise EvidenceError("Web discovery failed")

    with pytest.raises(EvidenceError, match="Web discovery failed"):
        run_job(setup, job, search_web=fail)
    job = store.get_job(job.run_id)
    assert job.stage == "gather" and job.search_calls == 1
    assert not job.searched_queries
    completed = run_job(setup, job, search_web=lambda *args: [])
    assert completed.search_calls == 2
    assert completed.model_calls == 3


def test_preflight_no_web_and_jev_are_explicit(setup):
    store, services, profile, _, _ = setup
    with pytest.raises(EvidenceError, match="Supply"):
        create_job(store, "Question", "chosen", profile, services, [], [], False)
    profile.reranker = Reranker.JEV
    with pytest.raises(Exception, match="Jev requires TYPESAFE_API_KEY"):
        make_job(setup)
    assert not store.runs()


def test_cli_research_resume_export_with_saved_model_choice(
    setup, runner, isolated_settings, monkeypatch, tmp_path
):
    store, services, profile, file, model = setup
    save_config(
        isolated_settings,
        AppConfig(default_profile="chosen", profiles={"chosen": profile}, services=services),
    )

    @contextmanager
    def open_index(settings):
        yield LocalIndex(store)

    monkeypatch.setattr(retrieval, "open_index", open_index)
    result = runner.invoke(
        app, ["research", "Compare retrieval", "--no-web", "--file", str(file), "--json"]
    )
    assert result.exit_code == 0, result.output
    job = json.loads(result.stdout)["research"]
    assert job["profile"]["model"] == "test-model"
    assert "plan (" in result.stderr
    result = runner.invoke(app, ["resume", job["run_id"], "--json"])
    assert result.exit_code == 0, result.output
    assert len(model.seen_tools) == 3
    result = runner.invoke(app, ["export", job["run_id"], "--output", str(tmp_path / "cli.md")])
    assert result.exit_code == 0, result.output


def test_prohibited_agent_tool_never_executes(setup):
    store, _, _, _, model = setup
    model.prohibited_tool = True
    job = make_job(setup)
    with pytest.raises(EvidenceError, match="prohibited tool"):
        run_job(setup, job)
    assert store.get_job(job.run_id).model_calls == 1
    assert len(model.seen_tools) == 1


def test_interrupt_is_saved_and_resume_uses_prior_plan(setup):
    store, _, _, _, _ = setup
    job = make_job(setup, web=True)

    def stop(*args):
        raise KeyboardInterrupt

    with pytest.raises(EvidenceError, match="Interrupted"):
        run_job(setup, job, search_web=stop)
    saved = store.get_job(job.run_id)
    assert saved.status == "interrupted" and saved.stage == "gather"
    assert saved.model_calls == 1
    assert run_job(setup, saved, search_web=lambda *args: []).model_calls == 3


def test_resume_rejects_changed_cluster_before_model_call(setup):
    store, services, _, _, model = setup
    job = make_job(setup)
    changed = services.model_copy(update={"elasticsearch_url": "https://other.example"})
    pipeline = ResearchPipeline(store, job, changed, LocalIndex(store), lambda msg: None)
    with store.lock(job.run_id), pytest.raises(EvidenceError, match="different Elasticsearch"):
        pipeline.run()
    assert not model.seen_tools


def test_unknown_passage_rejected(setup):
    from evidencelab.research import validate_claims
    from evidencelab.research_records import Citation, Claim

    citation = Citation(passage_id="0" * 64, quote="Unknown evidence quotation")
    with pytest.raises(EvidenceError, match="Citation validation failed"):
        validate_claims([Claim(text="Unverified", citations=[citation])], {})


def test_discovery_uses_urls_only_and_sanitizes_provider_failures(monkeypatch, setup):
    import tavily

    captured = {}

    class Client:
        def __init__(self, **kwargs):
            pass

        def search(self, **kwargs):
            captured.update(kwargs)
            return {
                "results": [
                    {"url": "https://docs.example/page", "content": "Unverified snippet"},
                    {"url": "https://docs.example/page#heading"},
                    {"url": "file:///private.txt"},
                ]
            }

    monkeypatch.setattr(tavily, "TavilyClient", Client)
    assert research.discover("test question", setup[1]) == ["https://docs.example/page"]
    assert captured["auto_parameters"] is False
    assert captured["include_answer"] is False
    assert captured["include_raw_content"] is False
    assert captured["max_results"] == 5

    def failure(*args, **kwargs):
        raise RuntimeError("API_SECRET")

    monkeypatch.setattr(Client, "search", failure)
    with pytest.raises(EvidenceError) as error:
        research.discover("test question", setup[1])
    assert "API_SECRET" not in str(error.value)


def test_cli_failure_keeps_json_clean_and_exposes_resume_id(
    setup, runner, isolated_settings, monkeypatch
):
    store, services, profile, file, model = setup
    save_config(
        isolated_settings,
        AppConfig(default_profile="chosen", profiles={"chosen": profile}, services=services),
    )
    model.fail_schema = "ResearchPlan"

    @contextmanager
    def open_index(settings):
        yield LocalIndex(store)

    monkeypatch.setattr(retrieval, "open_index", open_index)
    result = runner.invoke(
        app,
        [
            "research",
            "Question",
            "--no-web",
            "--file",
            str(file),
            "--json",
        ],
    )
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["run_id"] and data["stage"] == "plan" and data["status"] == "failed"
    assert "API_SECRET" not in result.output
    saved = store.get_job(data["run_id"])
    assert saved.model_calls == 1
    model.fail_schema = ""
    # Editing the default profile must not silently change a resumed run's model selection.
    save_config(isolated_settings, AppConfig(services=services))
    result = runner.invoke(app, ["resume", data["run_id"], "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["research"]["profile"]["model"] == "test-model"


def test_jev_scores_persist_and_resume_does_not_repeat_completed_reranking(setup, monkeypatch):
    from evidencelab.reranking import Relevance, RerankAudit, apply_ranking, ranking_key

    store, _, profile, _, model = setup
    profile.reranker = Reranker.JEV
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    calls = []

    def fake_rerank(question, hits, services, *, model, diagnostic=None):
        calls.append(question)
        audit = RerankAudit(
            cache_key=ranking_key(question, hits, model),
            model="jev-test",
            ratings=[
                Relevance(passage_id=h.passage.passage_id, score=3, confidence=1) for h in hits
            ],
        )
        return apply_ranking(hits, audit), audit

    monkeypatch.setattr(research, "rerank", fake_rerank)
    job = make_job(setup)
    model.fail_schema = "EvidenceReview"
    with pytest.raises(EvidenceError, match="model request failed"):
        run_job(setup, job)
    saved = store.get_job(job.run_id)
    assert saved.rerank_calls == 2 and len(saved.rerankings) == 2
    assert saved.stage == "review"
    model.fail_schema = ""
    finished = run_job(setup, saved)
    assert finished.status == "complete" and finished.rerank_calls == 2
    assert len(calls) == 2


def test_jev_failure_is_resumable_and_budget_is_persisted(setup, monkeypatch):
    store, _, profile, _, model = setup
    profile.reranker = Reranker.JEV
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def fail(*args, **kwargs):
        raise EvidenceError("Jev unavailable; no fallback")

    monkeypatch.setattr(research, "rerank", fail)
    job = make_job(setup)
    with pytest.raises(EvidenceError, match="no fallback"):
        run_job(setup, job)
    saved = store.get_job(job.run_id)
    assert saved.rerank_calls == 1 and saved.stage == "review"
    assert model.seen_tools == [["ResearchPlan"]]
    saved.rerank_calls = 6
    store.save_job(saved)
    with pytest.raises(EvidenceError, match="Jev request budget exhausted"):
        run_job(setup, saved)
    assert store.get_job(job.run_id).rerank_calls == 6


def test_untrusted_index_results_never_reach_jev(setup, monkeypatch):
    store, _, profile, _, _ = setup
    profile.reranker = Reranker.JEV
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    called = []
    monkeypatch.setattr(research, "rerank", lambda *args, **kwargs: called.append(True))
    index = LocalIndex(store)
    index.tamper = True
    with pytest.raises(EvidenceError, match="does not match"):
        run_job(setup, make_job(setup), index)
    assert not called
