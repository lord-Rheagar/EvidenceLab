import json

import pytest

from evidencelab.cli import app
from evidencelab.config import Depth
from evidencelab.evidence import EvidenceError
from evidencelab.ingestion import ingest_sources
from evidencelab.storage import EvidenceStore


@pytest.fixture
def store(tmp_path):
    value = EvidenceStore(tmp_path / "storage")
    yield value
    value.close()


def test_ingestion_deduplication_and_snapshot_persistence(store, tmp_path):
    first, duplicate = tmp_path / "first.txt", tmp_path / "copy.txt"
    first.write_text("A reliable evidence snapshot.")
    duplicate.write_text("A reliable evidence snapshot.")
    result = ingest_sources(store, [first, duplicate], [])
    run_id = result["run_id"]
    assert result["added_sources"] == 1
    assert result["duplicates"] == 1
    assert len(store.sources(run_id)[0].aliases) == 1
    first.write_text("Changed after ingestion.")
    again = ingest_sources(store, [first], [], run_id=run_id)
    assert again["duplicates"] == 1
    assert store.passages(run_id)[0].text == "A reliable evidence snapshot."
    other = EvidenceStore(store.path)
    try:
        assert other.passages(run_id) == store.passages(run_id)
    finally:
        other.close()


def test_run_isolation_and_unknown_runs(store, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("same document")
    first = ingest_sources(store, [path], [])["run_id"]
    second = ingest_sources(store, [path], [])["run_id"]
    assert store.passages(first)[0].passage_id != store.passages(second)[0].passage_id
    assert all(p.run_id == first for p in store.passages(first))
    with pytest.raises(EvidenceError, match="not found"):
        store.passages("unknown")


def test_oversized_input_rejected_before_run_creation(store, tmp_path):
    files = []
    for number in range(11):
        path = tmp_path / f"{number}.txt"
        path.write_text(str(number))
        files.append(path)
    with pytest.raises(EvidenceError, match="limit"):
        ingest_sources(store, files, [], depth=Depth.QUICK)
    assert store.runs() == []


def test_partial_ingestion_reports_skips(store, tmp_path):
    good, bad = tmp_path / "good.txt", tmp_path / "bad.pdf"
    good.write_text("useful text")
    bad.write_bytes(b"bad pdf")
    result = ingest_sources(store, [good, bad, tmp_path / "missing.txt"], [])
    assert result["added_sources"] == 1
    assert len(result["skipped"]) == 2


def test_run_lock_prevents_concurrent_mutation(store):
    run = store.create_run(Depth.THOROUGH)
    with store.lock(run.run_id), pytest.raises(EvidenceError, match="Another process"):
        with store.lock(run.run_id):
            pytest.fail("Second writer acquired lock")


def test_cli_local_ingest_inspect_and_runs(runner, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Search\n\nHybrid retrieval combines keyword and semantic search.")
    result = runner.invoke(app, ["ingest", "--file", str(path), "--local-only", "--json"])
    assert result.exit_code == 0, result.output
    result = json.loads(result.stdout)
    run_id = result["run_id"]
    assert result["passages"] == 1
    summary = runner.invoke(app, ["inspect", run_id, "--sources", "--json"])
    assert summary.exit_code == 0
    summary = json.loads(summary.stdout)
    assert summary["sources"][0]["title"] == "Search"
    listed = runner.invoke(app, ["runs", "--json"])
    assert json.loads(listed.stdout)["runs"][0]["run_id"] == run_id


def test_cli_index_requires_explicit_service_configuration(runner, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("test")
    result = runner.invoke(app, ["ingest", "--file", str(path), "--json"])
    assert result.exit_code == 2
    assert "Elasticsearch" in json.loads(result.stdout)["error"]


def test_cli_skipped_inputs_are_machine_readable(runner, tmp_path):
    result = runner.invoke(
        app, ["ingest", "--file", str(tmp_path / "missing.txt"), "--local-only", "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["skipped"]
