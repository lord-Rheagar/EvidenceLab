"""Run-scoped lexical and semantic retrieval through Elasticsearch."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ApiError, TransportError

from evidencelab.config import ConfigError, Services
from evidencelab.credentials import read_credential, service_slot
from evidencelab.evidence import EvidenceError, EvidenceRun, Passage, SearchHit, digest
from evidencelab.storage import EvidenceStore


def validate_search_settings(services: Services) -> str:
    if not services.elasticsearch_url or not services.inference_id:
        raise ConfigError(
            "Configure an Elasticsearch URL and embedding inference ID before indexing."
        )
    credential = read_credential(
        services.elasticsearch_api_key_env,
        service_slot("elasticsearch", services.elasticsearch_url),
    )
    if not credential.value:
        raise ConfigError("Set the Elasticsearch API key or store it with 'evidence configure'.")
    return credential.value


def index_name(inference_id: str) -> str:
    return f"evidencelab-passages-v1-{digest(inference_id)[:12]}"


def mappings(inference_id: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        name: {"type": "keyword"}
        for name in ["passage_id", "run_id", "source_id", "location", "locator", "published_at"]
    }
    properties.update({name: {"type": "integer"} for name in ["segment", "start", "end"]})
    properties.update(
        text={"type": "text"},
        title={"type": "text"},
        retrieved_at={"type": "date"},
        semantic_text={"type": "semantic_text", "inference_id": inference_id},
    )
    return {
        "dynamic": "strict",
        "_meta": {"evidencelab_schema": 1, "inference_id": inference_id},
        "properties": properties,
    }


def fuse(lexical: list[Passage], semantic: list[Passage], limit: int) -> list[SearchHit]:
    scores: dict[str, float] = defaultdict(float)
    passages: dict[str, Passage] = {}
    ranks: list[dict[str, int]] = []
    for result in (lexical, semantic):
        positions = {}
        for rank, passage in enumerate(result, 1):
            if passage.passage_id in positions:
                continue
            positions[passage.passage_id] = rank
            scores[passage.passage_id] += 1 / (60 + rank)
            passages[passage.passage_id] = passage
        ranks.append(positions)
    ordered = sorted(passages, key=lambda key: (-scores[key], key))
    # Give distinct sources a first slot, then fill remaining slots in fused rank order.
    selected: list[str] = []
    seen_sources: set[str] = set()
    for key in ordered:
        if passages[key].source_id not in seen_sources:
            selected.append(key)
            seen_sources.add(passages[key].source_id)
    selected.extend(key for key in ordered if key not in selected)
    return [
        SearchHit(
            passage=passages[key],
            score=scores[key],
            lexical_rank=ranks[0].get(key),
            semantic_rank=ranks[1].get(key),
        )
        for key in selected[:limit]
    ]


class EvidenceIndex:
    def __init__(self, client: Elasticsearch, services: Services) -> None:
        self.client = client
        self.services = services
        self.inference_id = services.inference_id or ""
        self.name = index_name(self.inference_id)

    def check_binding(self, run: EvidenceRun) -> None:
        if run.index_name and (
            run.index_name != self.name
            or run.elasticsearch_url != self.services.elasticsearch_url
            or run.inference_id != self.inference_id
        ):
            raise EvidenceError(
                "This run uses a different cluster or embedding endpoint. "
                "Restore its settings or create a new run."
            )

    def check_endpoint(self) -> None:
        response = self.client.inference.get(inference_id=self.inference_id)
        endpoints = response.get("endpoints", [])
        if not any(
            item.get("inference_id") == self.inference_id
            and item.get("task_type") in {"text_embedding", "sparse_embedding", "embedding"}
            for item in endpoints
        ):
            raise EvidenceError("The configured inference endpoint is not an embedding endpoint.")

    def ensure_index(self) -> None:
        self.check_endpoint()
        if not self.client.indices.exists(index=self.name):
            try:
                self.client.indices.create(index=self.name, mappings=mappings(self.inference_id))
            except ApiError as exc:
                if exc.error != "resource_already_exists_exception":
                    raise
        self.validate_mapping()

    def validate_mapping(self) -> None:
        response = self.client.indices.get_mapping(index=self.name)
        actual = response[self.name]["mappings"]
        expected = mappings(self.inference_id)
        semantic = actual.get("properties", {}).get("semantic_text", {})
        if (
            actual.get("_meta") != expected["_meta"]
            or semantic.get("inference_id") != self.inference_id
            or any(
                actual.get("properties", {}).get(field, {}).get("type") != definition["type"]
                for field, definition in expected["properties"].items()
            )
        ):
            raise EvidenceError(
                "Existing evidence index has incompatible metadata; "
                "use a new embedding endpoint/index."
            )

    def index_run(self, store: EvidenceStore, run_id: str) -> int:
        run = store.get_run(run_id)
        self.check_binding(run)
        passages = store.passages(run_id)
        if not passages:
            raise EvidenceError("This run has no passages to index.")
        self.ensure_index()
        run.index_name, run.elasticsearch_url, run.inference_id = (
            self.name,
            self.services.elasticsearch_url,
            self.inference_id,
        )
        run.status = "indexing"
        store.save_run(run)
        try:
            for start in range(0, len(passages), 50):
                operations: list[dict[str, Any]] = []
                batch = passages[start : start + 50]
                for passage in batch:
                    operations.append({"index": {"_id": passage.passage_id}})
                    operations.append(
                        {**passage.model_dump(mode="json"), "semantic_text": passage.text}
                    )
                response = self.client.bulk(index=self.name, operations=operations)
                items = response.get("items", [])
                if (
                    response.get("errors")
                    or len(items) != len(batch)
                    or any(item.get("index", {}).get("status") not in {200, 201} for item in items)
                ):
                    raise EvidenceError(
                        "Some passages failed to index. Snapshots are saved; "
                        "retry 'evidence index RUN_ID'."
                    )
            self.client.indices.refresh(index=self.name)
        except (ApiError, TransportError, EvidenceError):
            run.status = "index_failed"
            store.save_run(run)
            raise
        run.status = "indexed"
        store.save_run(run)
        return len(passages)

    def search(self, run: EvidenceRun, question: str, limit: int = 12) -> list[SearchHit]:
        if not question.strip() or len(question) > 8000:
            raise EvidenceError("Provide a nonempty question of at most 8,000 characters.")
        if not 1 <= limit <= 30:
            raise EvidenceError("Search limit must be between 1 and 30.")
        self.check_binding(run)
        if run.status != "indexed":
            raise EvidenceError("Run is not fully indexed. Run 'evidence index RUN_ID' first.")
        self.validate_mapping()
        lists = []
        for clause in (
            {"multi_match": {"query": question, "fields": ["text", "title^2"]}},
            {"match": {"semantic_text": {"query": question}}},
        ):
            response = self.client.search(
                index=self.name,
                query={"bool": {"filter": [{"term": {"run_id": run.run_id}}], "must": [clause]}},
                size=30,
                source_excludes=["semantic_text"],
                allow_partial_search_results=False,
            )
            if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
                raise EvidenceError("Search did not complete; retry the query.")
            passages = [Passage.model_validate(hit["_source"]) for hit in response["hits"]["hits"]]
            if any(p.run_id != run.run_id for p in passages):
                raise EvidenceError(
                    "Search returned evidence outside the selected run; results rejected."
                )
            lists.append(passages)
        return fuse(lists[0], lists[1], limit)


@contextmanager
def open_index(services: Services) -> Iterator[EvidenceIndex]:
    key = validate_search_settings(services)
    client = Elasticsearch(
        services.elasticsearch_url, api_key=key, request_timeout=60, max_retries=0
    )
    try:
        yield EvidenceIndex(client, services)
    except (ApiError, TransportError):
        raise EvidenceError(
            "Elasticsearch request failed. Check connectivity, key privileges, "
            "and the embedding endpoint."
        ) from None
    finally:
        client.close()
