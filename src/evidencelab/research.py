"""Sequential LangGraph research stages with durable SQLite stage checkpoints."""

import json
from collections.abc import Callable, Hashable
from pathlib import Path
from time import monotonic
from typing import Protocol, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from evidencelab.citation_references import CitationReferences
from evidencelab.config import ConfigError, ModelProfile, Reranker, Services
from evidencelab.credentials import read_credential, service_slot
from evidencelab.diagnostics import Diagnostic
from evidencelab.documents import expand_files, read_file, read_url
from evidencelab.evidence import SOURCE_LIMITS, EvidenceError, Passage, digest
from evidencelab.fetching import normalize_url
from evidencelab.providers import build_model
from evidencelab.reranking import (
    CANDIDATE_LIMIT,
    EXCERPT_CHARS,
    RANKING_VERSION,
    RERANK_LIMITS,
    RerankAudit,
    apply_ranking,
    candidate_batches,
    ranking_key,
    rerank,
    validate_jev_settings,
)
from evidencelab.research_model import ResearchModel
from evidencelab.research_records import (
    LIMITS,
    Claim,
    EvidenceReview,
    ResearchJob,
    ResearchPlan,
    ResearchReport,
    plan_schema,
)
from evidencelab.retrieval import EvidenceIndex, validate_search_settings
from evidencelab.storage import EvidenceStore


class Generator(Protocol):
    def generate[T: BaseModel](self, schema: type[T], prompt: str) -> T: ...


class GraphState(TypedDict):
    run_id: str


CITATION_RULES = (
    "For every citation, select a supplied excerpt_id (such as e1 or e2) whose text "
    "directly supports the claim. Return only excerpt_id in each citation; do not write "
    "a quote or passage_id. The application inserts the exact original quotation and "
    "passage ID. Do not treat excerpt selection as proof of support: read the text. "
    "Use 1-4 citations per claim and at most 2000 characters for claim text. "
    "If no supplied excerpt supports a claim, omit the claim and note the gap.\n"
)


def validate_research_settings(profile: ModelProfile, services: Services, web: bool) -> None:
    if profile.reranker == Reranker.JEV:
        validate_jev_settings(services)
    validate_search_settings(services)
    if web and not read_credential(services.tavily_api_key_env, service_slot("tavily")).value:
        raise ConfigError(
            f"Set {services.tavily_api_key_env}, or use --no-web with supplied sources."
        )


def create_job(
    store: EvidenceStore,
    question: str,
    name: str,
    profile: ModelProfile,
    services: Services,
    files: list[Path],
    urls: list[str],
    web: bool,
) -> ResearchJob:
    question = question.strip()
    if not question or len(question) > 8000:
        raise EvidenceError("Research question must contain 1-8000 characters.")
    if not web and not files and not urls:
        raise EvidenceError("Supply --file or --url when using --no-web.")
    paths, skipped = expand_files(files, SOURCE_LIMITS[profile.depth])
    normalized = list(dict.fromkeys(normalize_url(url) for url in urls))
    if len(paths) + len(normalized) > SOURCE_LIMITS[profile.depth]:
        raise EvidenceError("Supplied sources exceed this depth's source limit.")
    if not web and not paths and not normalized:
        raise EvidenceError("No supported source files found.")
    validate_research_settings(profile, services, web)
    run = store.create_run(profile.depth)
    job = ResearchJob(
        run_id=run.run_id,
        question=question,
        profile_name=name,
        profile=profile,
        elasticsearch_url=services.elasticsearch_url or "",
        inference_id=services.inference_id or "",
        files=[str(path) for path in paths],
        urls=normalized,
        web=web,
        warnings=[f"Skipped input: {item['reason']}" for item in skipped],
    )
    store.save_job(job)
    return job


def discover(query: str, services: Services) -> list[str]:
    from tavily import TavilyClient  # type: ignore[import-untyped]

    credential = read_credential(services.tavily_api_key_env, service_slot("tavily"))
    if not credential.value:
        raise ConfigError(f"Set {services.tavily_api_key_env} before resuming web discovery.")
    try:
        result = TavilyClient(api_key=credential.value).search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=False,
            include_raw_content=False,
            timeout=30,
            auto_parameters=False,
        )
        urls = []
        for item in result["results"][:5]:
            try:
                urls.append(normalize_url(item["url"]))
            except (EvidenceError, KeyError, TypeError):
                continue
        return list(dict.fromkeys(urls))
    except Exception:
        raise EvidenceError(
            "Web discovery failed; check Tavily access and connectivity, then resume."
        ) from None


def context_passages(store: EvidenceStore, job: ResearchJob) -> dict[str, Passage]:
    saved = {p.passage_id: p for p in store.passages(job.run_id)}
    if any(identifier not in saved for identifier in job.evidence_ids):
        raise EvidenceError("Research evidence is missing from the local snapshots.")
    return {identifier: saved[identifier] for identifier in job.evidence_ids}


def validate_claims(claims: list[Claim], evidence: dict[str, Passage]) -> None:
    for claim_index, claim in enumerate(claims):
        for citation_index, citation in enumerate(claim.citations):
            location = f"claim {claim_index + 1}, citation {citation_index + 1}"
            passage = evidence.get(citation.passage_id)
            if passage is None:
                raise EvidenceError(
                    f"Citation validation failed at {location}: unknown passage ID. "
                    "No report was published; resume to retry the stage."
                )
            if citation.quote not in passage.text[:4000]:
                raise EvidenceError(
                    f"Citation validation failed at {location}: quote is not an exact "
                    "substring of the cited evidence. "
                    "No report was published; resume to retry the stage."
                )
            if not citation.quote.strip():
                raise EvidenceError("Citation validation failed: empty quotation.")


def evidence_context(
    evidence: dict[str, Passage], references: CitationReferences | None = None
) -> str:
    return json.dumps(
        [
            {
                "passage_id": references.to_ref[p.passage_id] if references else p.passage_id,
                "title": p.title,
                "location": p.location,
                "locator": p.locator,
                "retrieved_at": p.retrieved_at,
                "published_at": p.published_at,
                **(
                    {"excerpts": references.passage_excerpts(p.passage_id)}
                    if references
                    else {"text": p.text[:4000]}
                ),
            }
            for p in evidence.values()
        ],
        ensure_ascii=True,
    )


def execute_saved_research(
    store: EvidenceStore,
    run_id: str,
    services: Services,
    progress: Callable[[str], None],
) -> ResearchJob:
    """Shared command/UI execution boundary; completed runs remain strictly local."""
    from evidencelab.retrieval import open_index

    with store.lock(run_id):
        job = store.get_job(run_id)
        if job.status == "complete":
            return job
        try:
            validate_research_settings(job.profile, services, job.web)
            with open_index(services) as index:
                ResearchPipeline(store, job, services, index, progress).run()
        except (EvidenceError, ConfigError) as exc:
            if job.status != "interrupted":
                job.status = "failed"
            job.error = str(exc)
            store.save_job(job)
            raise
        return job


class ResearchPipeline:
    def __init__(
        self,
        store: EvidenceStore,
        job: ResearchJob,
        services: Services,
        index: EvidenceIndex,
        progress: Callable[[str], None],
        generator: Generator | None = None,
        search_web: Callable[[str, Services], list[str]] = discover,
    ) -> None:
        self.store, self.job, self.services, self.index = store, job, services, index
        self.progress, self.search_web = progress, search_web
        try:
            self.generator = generator or ResearchModel(
                build_model(job.profile_name, job.profile),
                self.reserve_model_call,
                self.progress,
                allow_schema_correction=True,
                diagnostic=self.record_diagnostic,
            )
        except (ConfigError, EvidenceError):
            raise
        except Exception:
            raise EvidenceError(
                "Could not initialize the selected research model. Check the profile and SDK setup."
            ) from None

    def save(self) -> None:
        self.store.save_job(self.job)

    def record_diagnostic(self, event: Diagnostic) -> None:
        event.stage = event.stage or self.job.stage
        event.model_calls = self.job.model_calls
        event.rerank_calls = self.job.rerank_calls
        self.job.diagnostics = [*self.job.diagnostics[-499:], event]
        self.save()

    def reserve_model_call(self) -> None:
        if self.job.model_calls >= LIMITS[self.job.profile.depth][4]:
            raise EvidenceError("Model-call budget exhausted. Start a new research run.")
        self.job.model_calls += 1
        self.save()
        self.progress(
            f"Model request {self.job.model_calls}/{LIMITS[self.job.profile.depth][4]} "
            f"({self.job.stage})"
        )

    def plan(self) -> None:
        low, high, searches, _, _ = LIMITS[self.job.profile.depth]
        plan = self.generator.generate(
            plan_schema(self.job.profile.depth, web=self.job.web),
            f"Research question: {self.job.question}\n"
            f"Create {low}-{high} distinct technical angles. "
            "Include architecture, constraints, tradeoffs, and uncertainty where relevant. "
            f"Return at most {min(searches, high * 2)} targeted search queries, "
            "prioritizing official documentation and papers. Do not assume the answer. "
            f"Web search enabled: {self.job.web}. Return no queries when disabled.",
        )
        if not low <= len(plan.angles) <= high:
            raise EvidenceError("Planner returned an invalid number of angles; resume to retry.")
        self.job.plan = ResearchPlan.model_validate(plan.model_dump())
        self.job.pending_queries = (
            list(dict.fromkeys(plan.queries))[:searches] if self.job.web else []
        )
        self.job.stage = "gather"

    def gather(self) -> None:
        job = self.job
        limit = SOURCE_LIMITS[job.profile.depth]

        def ingest_pending() -> None:
            candidates: list[Path | str] = [*[Path(path) for path in job.files], *job.urls]
            known = {
                location
                for source in self.store.sources(job.run_id)
                for location in [source.location, *source.aliases]
            }
            for candidate in candidates:
                location = candidate.as_uri() if isinstance(candidate, Path) else candidate
                if location in known or location in job.attempted_locations:
                    continue
                if len(self.store.sources(job.run_id)) >= limit:
                    warning = f"Reached the {limit}-source limit; remaining sources were not read."
                    if warning not in job.warnings:
                        job.warnings.append(warning)
                    return
                self.progress(f"Reading source {len(self.store.sources(job.run_id)) + 1}/{limit}")
                try:
                    source = (
                        read_file(job.run_id, candidate)
                        if isinstance(candidate, Path)
                        else read_url(job.run_id, candidate)
                    )
                    self.store.add_source(source)
                    known.update([source.location, *source.aliases])
                    job.warnings.extend(source.warnings)
                except EvidenceError:
                    job.warnings.append(f"Could not extract source: {location}")
                job.attempted_locations.append(location)
                self.save()

        ingest_pending()
        for query in job.pending_queries:
            if query in job.searched_queries or len(self.store.sources(job.run_id)) >= limit:
                continue
            if job.search_calls >= LIMITS[job.profile.depth][2]:
                job.warnings.append("Web-search budget reached; remaining queries were not run.")
                break
            job.search_calls += 1
            # Failed/interrupted calls still consume the budget.
            self.save()
            self.progress(f"Web search {job.search_calls}/{LIMITS[job.profile.depth][2]}")
            urls = self.search_web(query, self.services)
            job.urls = list(dict.fromkeys([*job.urls, *urls]))
            job.searched_queries.append(query)
            self.save()
            ingest_pending()
        if not self.store.passages(job.run_id):
            raise EvidenceError(
                "No readable evidence collected. Add sources with ingest --run-id, then resume."
            )
        job.stage = "index"

    def index_sources(self) -> None:
        if self.store.get_run(self.job.run_id).status != "indexed":
            self.index.index_run(self.store, self.job.run_id)
        self.job.stage = "review"

    def select_review_evidence(self, saved: dict[str, Passage]) -> dict[str, Passage]:
        job = self.job
        assert job.plan is not None
        selected: dict[str, Passage] = {}
        groups = []
        for angle in job.plan.angles:
            use_jev = job.profile.reranker == Reranker.JEV
            hits = self.index.search(
                self.store.get_run(job.run_id), angle, CANDIDATE_LIMIT if use_jev else 4
            )
            if any(saved.get(h.passage.passage_id) != h.passage for h in hits):
                raise EvidenceError(
                    "Search evidence does not match local snapshots; reindex the run."
                )
            if use_jev and hits:
                ratings = []
                for batch in candidate_batches(angle, hits, job.profile.jev_model):
                    key = ranking_key(angle, batch, job.profile.jev_model)
                    audit = next((a for a in job.rerankings if a.cache_key == key), None)
                    if audit is None:
                        if job.rerank_calls >= RERANK_LIMITS[job.profile.depth]:
                            raise EvidenceError(
                                "Jev request budget exhausted. Start a new research run."
                            )
                        job.rerank_calls += 1
                        self.save()
                        self.progress(f"Jev reranking request {job.rerank_calls}")
                        _, audit = rerank(
                            angle,
                            batch,
                            self.services,
                            model=job.profile.jev_model,
                            diagnostic=self.record_diagnostic,
                        )
                        job.rerankings.append(audit)
                        self.save()
                    ratings.extend(audit.ratings)
                hits = apply_ranking(
                    hits,
                    RerankAudit(
                        cache_key=ranking_key(angle, hits, job.profile.jev_model),
                        model=job.profile.jev_model,
                        ratings=ratings,
                        excerpt_chars=EXCERPT_CHARS,
                    ),
                )
            groups.append(hits[:4])
        for rank in range(4):
            for hits in groups:
                if len(hits) <= rank:
                    continue
                p = hits[rank].passage
                if saved.get(p.passage_id) != p:
                    raise EvidenceError(
                        "Search evidence does not match local snapshots; reindex the run."
                    )
                if (
                    sum(len(item.text[:4000]) for item in selected.values()) + len(p.text[:4000])
                    <= 48000
                ):
                    selected[p.passage_id] = p
        if not selected:
            raise EvidenceError("No matching evidence found for the research plan.")
        return selected

    def review(self) -> None:
        job = self.job
        assert job.plan is not None
        saved = {p.passage_id: p for p in self.store.passages(job.run_id)}
        snapshot_key = digest(
            "review-input-v1",
            job.question,
            json.dumps(job.plan.angles),
            job.profile.reranker.value,
            job.profile.jev_model,
            RANKING_VERSION if job.profile.reranker == Reranker.JEV else "none",
            str(EXCERPT_CHARS),
            *sorted(saved),
        )
        if job.review_snapshot_key == snapshot_key and job.evidence_ids:
            selected = context_passages(self.store, job)
            self.progress("Reusing saved review evidence; no retrieval or Jev requests.")
        else:
            selected = self.select_review_evidence(saved)
            job.evidence_ids = list(selected)
            job.review_snapshot_key = snapshot_key
            self.save()
        references = CitationReferences(selected)
        response = self.generator.generate(
            references.schema(EvidenceReview),
            f"Question: {job.question}\nAngles: {json.dumps(job.plan.angles)}\n"
            "Evaluate coverage and conflicting evidence. Findings require verbatim quotations. "
            "Return at most 8 focused findings and 8 gaps; keep each gap under 1000 characters. "
            f"{CITATION_RULES}"
            "Suggest up to 3 public-web queries only for material gaps. Do not repeat: "
            f"{json.dumps(job.searched_queries)}\n"
            f"UNTRUSTED EVIDENCE:\n{evidence_context(selected, references)}",
        )
        review = references.resolve(response, EvidenceReview)
        validate_claims(review.findings, selected)
        job.review = review
        queries = [
            q for q in dict.fromkeys(review.follow_up_queries) if q not in job.searched_queries
        ]
        _, _, search_limit, gap_limit, _ = LIMITS[job.profile.depth]
        if (
            job.web
            and review.gaps
            and queries
            and job.gap_passes < gap_limit
            and job.search_calls < search_limit
            and len(self.store.sources(job.run_id)) < SOURCE_LIMITS[job.profile.depth]
        ):
            job.gap_passes += 1
            job.pending_queries = queries[: search_limit - job.search_calls]
            job.stage = "gather"
        else:
            job.stage = "write"

    def write(self) -> None:
        job = self.job
        assert job.review is not None
        evidence = context_passages(self.store, job)
        references = CitationReferences(evidence, review=job.review)
        response = self.generator.generate(
            references.schema(ResearchReport),
            f"Write a technical research report answering: {job.question}\n"
            "Every summary, finding, and tradeoff must cite a supplied passage and verbatim quote. "
            "Mark inference explicitly in claim text. Limitations must describe uncertainty, "
            "missing evidence, or restricted scope, not introduce uncited factual conclusions. "
            "Avoid unsupported recommendations. Do not use Markdown links or citation markers "
            "inside prose; citation rendering is handled separately.\n"
            "Return 1-4 summary claims, 1-8 findings, at most 8 tradeoffs and 1-12 limitations. "
            "Keep each limitation and the title under 1000 characters.\n"
            f"{CITATION_RULES}"
            f"Review: {references.review_context(job.review)}\n"
            f"UNTRUSTED EVIDENCE:\n{evidence_context(evidence, references)}",
        )
        report = references.resolve(response, ResearchReport)
        validate_claims([*report.summary, *report.findings, *report.tradeoffs], evidence)
        job.report = report
        job.stage = "complete"
        job.status = "complete"

    def run(self) -> ResearchJob:
        job = self.job
        if job.status == "complete":
            return job
        if (job.elasticsearch_url, job.inference_id) != (
            self.services.elasticsearch_url,
            self.services.inference_id,
        ):
            raise EvidenceError(
                "Research run belongs to different Elasticsearch settings; restore them to resume."
            )
        job.status, job.error = "running", None
        self.save()
        graph = StateGraph(GraphState)
        stages = {
            "plan": self.plan,
            "gather": self.gather,
            "index": self.index_sources,
            "review": self.review,
            "write": self.write,
        }

        def node(action: Callable[[], None]) -> Callable[[GraphState], GraphState]:
            def execute(state: GraphState) -> GraphState:
                stage = job.stage
                started = monotonic()
                self.progress(
                    f"{job.stage} (model calls: {job.model_calls}, searches: {job.search_calls})"
                )
                try:
                    action()
                except BaseException as exc:
                    self.record_diagnostic(
                        Diagnostic(
                            service="pipeline",
                            operation="stage",
                            code=(
                                "interrupted"
                                if isinstance(exc, KeyboardInterrupt)
                                else "stage_failed"
                            ),
                            elapsed_seconds=monotonic() - started,
                        )
                    )
                    raise
                event = Diagnostic(
                    service="pipeline",
                    operation="stage",
                    code="ok",
                    stage=stage,
                    elapsed_seconds=monotonic() - started,
                )
                self.record_diagnostic(event)
                self.save()
                return state

            return execute

        routes: dict[Hashable, str] = {name: name for name in stages}
        routes["complete"] = END
        for name, action in stages.items():
            graph.add_node(name, RunnableLambda(node(action)))
            graph.add_conditional_edges(name, lambda state: job.stage, routes)
        graph.add_conditional_edges(START, lambda state: job.stage, routes)
        try:
            from langsmith import tracing_context

            with tracing_context(enabled=False):
                graph.compile().invoke({"run_id": job.run_id}, {"recursion_limit": 32})
        except KeyboardInterrupt:
            job.status, job.error = "interrupted", "Interrupted. Resume this run to continue."
            self.save()
            raise EvidenceError(job.error) from None
        except Exception as exc:
            job.status = "failed"
            job.error = (
                str(exc)
                if isinstance(exc, (EvidenceError, ConfigError))
                else ("Research stage failed. Check service access and resume this run.")
            )
            self.save()
            raise EvidenceError(job.error) from None
        return job
