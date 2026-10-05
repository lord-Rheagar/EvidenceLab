import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from evidencelab import retrieval
from evidencelab.cli import app
from evidencelab.config import Services
from evidencelab.evidence import EvidenceError, SearchHit
from evidencelab.ingestion import ingest_sources
from evidencelab.retrieval import EvidenceIndex, fuse
from evidencelab.storage import EvidenceStore


class FakeCluster:
    def __init__(self):
        self.mapping = {}
        self.documents = {}
        self.queries = []
        self.fail_bulk = False
        self.leak_run = False
        self.refresh_count = 0
        self.inference = SimpleNamespace(
            get=lambda **kw: {
                "endpoints": [{"inference_id": kw["inference_id"], "task_type": "text_embedding"}]
            }
        )
        self.indices = SimpleNamespace(
            exists=lambda **kw: kw["index"] in self.mapping,
            create=self.create,
            get_mapping=lambda **kw: {kw["index"]: {"mappings": self.mapping[kw["index"]]}},
            refresh=self.refresh,
        )

    def create(self, *, index, mappings):
        self.mapping[index] = copy.deepcopy(mappings)

    def refresh(self, **kwargs):
        self.refresh_count += 1

    def bulk(self, *, index, operations):
        items = []
        for action, document in zip(operations[::2], operations[1::2], strict=True):
            self.documents[action["index"]["_id"]] = copy.deepcopy(document)
            status = 500 if self.fail_bulk else 201
            items.append({"index": {"status": status}})
        return {"errors": self.fail_bulk, "items": items}

    def search(self, **kwargs):
        self.queries.append(kwargs)
        run_id = kwargs["query"]["bool"]["filter"][0]["term"]["run_id"]
        values = [document for document in self.documents.values() if document["run_id"] == run_id]
        if self.leak_run:
            values = list(self.documents.values())
        return {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            key: value for key, value in document.items() if key != "semantic_text"
                        }
                    }
                    for document in values
                ]
            }
        }


@pytest.fixture
def setup(tmp_path):
    store = EvidenceStore(tmp_path / "index-test")
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("Keyword search supports exact identifiers.")
    second.write_text("Semantic search retrieves related concepts.")
    run_id = ingest_sources(store, [first, second], [])["run_id"]
    cluster = FakeCluster()
    services = Services(elasticsearch_url="https://elastic.example", inference_id="embedding-test")
    index = EvidenceIndex(cluster, services)
    yield store, run_id, cluster, index
    store.close()


def test_indexing_retry_is_idempotent_and_search_is_run_scoped(setup):
    store, run_id, cluster, index = setup
    assert index.index_run(store, run_id) == 2
    assert index.index_run(store, run_id) == 2
    assert len(cluster.documents) == 2
    assert cluster.refresh_count == 2
    run = store.get_run(run_id)
    assert run.status == "indexed"
    results = index.search(run, "retrieval")
    assert len(results) == 2
    assert len(cluster.queries) == 2
    for query in cluster.queries:
        assert query["query"]["bool"]["filter"] == [{"term": {"run_id": run_id}}]
        assert query["allow_partial_search_results"] is False
    assert all(p.passage.text in {item.text for item in store.passages(run_id)} for p in results)


def test_failed_bulk_preserves_snapshots_and_can_resume(setup):
    store, run_id, cluster, index = setup
    cluster.fail_bulk = True
    with pytest.raises(EvidenceError, match="failed to index"):
        index.index_run(store, run_id)
    assert store.get_run(run_id).status == "index_failed"
    assert len(store.passages(run_id)) == 2
    with pytest.raises(EvidenceError, match="not fully indexed"):
        index.search(store.get_run(run_id), "retrieval")
    cluster.fail_bulk = False
    assert index.index_run(store, run_id) == 2
    assert len(cluster.documents) == 2
    assert store.get_run(run_id).status == "indexed"


def test_cluster_and_embedding_binding_are_enforced(setup):
    store, run_id, cluster, index = setup
    index.index_run(store, run_id)
    wrong = EvidenceIndex(
        cluster, Services(elasticsearch_url="https://other.example", inference_id="other-model")
    )
    with pytest.raises(EvidenceError, match="different cluster"):
        wrong.index_run(store, run_id)
    assert len(cluster.mapping) == 1


def test_bad_existing_mapping_and_non_embedding_endpoint_are_rejected(setup):
    store, run_id, cluster, index = setup
    index.ensure_index()
    cluster.mapping[index.name]["_meta"]["inference_id"] = "wrong"
    with pytest.raises(EvidenceError, match="incompatible"):
        index.index_run(store, run_id)
    cluster.inference.get = lambda **kw: {
        "endpoints": [{"inference_id": index.inference_id, "task_type": "completion"}]
    }
    with pytest.raises(EvidenceError, match="not an embedding"):
        index.check_endpoint()


def test_leaked_run_is_rejected_even_if_cluster_ignores_filter(setup, tmp_path):
    store, run_id, cluster, index = setup
    index.index_run(store, run_id)
    path = tmp_path / "private.txt"
    path.write_text("Other run's information")
    other = ingest_sources(store, [path], [])["run_id"]
    index.index_run(store, other)
    cluster.leak_run = True
    with pytest.raises(EvidenceError, match="outside the selected run"):
        index.search(store.get_run(run_id), "retrieval")


def test_reciprocal_rank_fusion_and_duplicate_candidates(setup):
    store, run_id, _, _ = setup
    first, second = store.passages(run_id)
    results = fuse([first, second, second], [second], 2)
    assert results[0].passage == second
    assert results[0].lexical_rank == 2
    assert results[0].semantic_rank == 1
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_search_rejects_timeout_instead_of_silently_using_partial_results(setup):
    store, run_id, cluster, index = setup
    index.index_run(store, run_id)
    cluster.search = lambda **kw: {"timed_out": True}
    with pytest.raises(EvidenceError, match="did not complete"):
        index.search(store.get_run(run_id), "retrieval")


def test_cli_full_ingest_index_search_flow(runner, monkeypatch, tmp_path):
    cluster = FakeCluster()
    services = Services(elasticsearch_url="https://elastic.example", inference_id="embedding-test")
    monkeypatch.setenv("ELASTICSEARCH_URL", services.elasticsearch_url)
    monkeypatch.setenv("EVIDENCE_INFERENCE_ID", services.inference_id)
    monkeypatch.setenv("ELASTICSEARCH_API_KEY", "test-key")

    @contextmanager
    def mocked_index(settings):
        yield EvidenceIndex(cluster, settings)

    monkeypatch.setattr(retrieval, "open_index", mocked_index)
    path = tmp_path / "notes.txt"
    path.write_text("Index passages with stable citation identifiers.")
    result = runner.invoke(app, ["ingest", "--file", str(path), "--json"])
    assert result.exit_code == 0, result.output
    run = json.loads(result.stdout)
    assert run["status"] == "indexed"
    result = runner.invoke(app, ["search", run["run_id"], "citations", "--json"])
    assert result.exit_code == 0, result.output
    hit = json.loads(result.stdout)["hits"][0]["passage"]
    assert hit["location"] == path.as_uri()
    result = runner.invoke(
        app, ["inspect", run["run_id"], "--passage", hit["passage_id"], "--json"]
    )
    assert json.loads(result.stdout)["passage"]["text"] == hit["text"]


def test_cli_rejects_tampered_remote_citation(runner, monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Original content")
    result = runner.invoke(app, ["ingest", "--file", str(path), "--local-only", "--json"])
    run_id = json.loads(result.stdout)["run_id"]
    store = EvidenceStore()
    try:
        original = store.passages(run_id)[0]
    finally:
        store.close()
    tampered = original.model_copy(update={"text": "Invented remote content"})

    @contextmanager
    def mocked_index(settings):
        yield SimpleNamespace(search=lambda *args: [SearchHit(passage=tampered, score=1)])

    monkeypatch.setattr(retrieval, "open_index", mocked_index)
    result = runner.invoke(app, ["search", run_id, "citations", "--json"])
    assert result.exit_code == 1
    assert "does not match" in json.loads(result.stdout)["error"]
    assert "Invented remote content" not in result.output
