import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from evidencelab.documents import expand_files, extract_source, read_file
from evidencelab.evidence import EvidenceError, chunk_source


def pdf_bytes(*, encrypted=False, blank=False):
    writer = PdfWriter()
    for text in ("First page evidence", "Second page evidence"):
        page = writer.add_blank_page(width=612, height=792)
        if not blank:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = stream
    if encrypted:
        writer.encrypt("test-password")
    target = io.BytesIO()
    writer.write(target)
    return target.getvalue()


def test_markdown_chunk_citations_are_exact(tmp_path):
    path = tmp_path / "notes.md"
    text = "# Deployment\n\n" + "\n\n".join(
        " ".join(f"word{n}" for n in range(250)) for _ in range(8)
    )
    path.write_text(text, encoding="utf-8")
    source = read_file("run", path)
    assert source.title == "Deployment"
    chunks = chunk_source(source)
    assert len(chunks) > 3
    assert all(len(p.text.split()) <= 600 for p in chunks)
    assert all(p.text == source.segments[p.segment].text[p.start : p.end] for p in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(source.segments[0].text)
    assert all(left.end >= right.start for left, right in zip(chunks, chunks[1:], strict=False))
    assert len({p.passage_id for p in chunks}) == len(chunks)
    assert "lines" in chunks[0].locator


def test_pdf_retains_page_citations():
    source = extract_source("run", "file:///paper.pdf", pdf_bytes(), "pdf", "Paper")
    chunks = chunk_source(source)
    assert [p.locator for p in chunks] == ["PDF page 1", "PDF page 2"]
    assert "First page evidence" in chunks[0].text
    assert "Second page evidence" in chunks[1].text


@pytest.mark.parametrize(
    "options,reason", [({"encrypted": True}, "Encrypted"), ({"blank": True}, "no extractable text")]
)
def test_unsupported_pdf_reports_reason(options, reason):
    with pytest.raises(EvidenceError, match=reason):
        extract_source("run", "file:///paper.pdf", pdf_bytes(**options), "pdf", "Paper")


def test_html_title_and_content():
    article = "Elasticsearch supports structured filters and lexical retrieval. " * 20
    html = (
        "<html><head><title>Search Architecture</title></head><body><article>"
        f"<h1>Search Architecture</h1><p>{article}</p></article></body></html>"
    )
    source = extract_source("run", "https://example.com/article", html.encode(), "html", "fallback")
    assert source.title == "Search Architecture"
    assert "structured filters" in source.segments[0].text
    assert "Extracted article" in chunk_source(source)[0].locator


def test_limits_invalid_encoding_and_corrupt_pdf(monkeypatch):
    from evidencelab import documents

    monkeypatch.setattr(documents, "MAX_BYTES", 10)
    with pytest.raises(EvidenceError, match="20 MB"):
        extract_source("r", "x", b"a" * 11, "text", "x")
    with pytest.raises(EvidenceError, match="parsed"):
        extract_source("r", "x", b"\xff", "text", "x")
    with pytest.raises(EvidenceError, match="parsed"):
        extract_source("r", "x", b"invalid", "pdf", "x")


def test_folder_input_deduplicates_and_skips_hidden(tmp_path):
    (tmp_path / "one.md").write_text("one")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.txt").write_text("hidden")
    (tmp_path / "image.png").write_bytes(b"png")
    paths, skipped = expand_files([tmp_path, tmp_path / "one.md"], 10)
    assert paths == [(tmp_path / "one.md").resolve()]
    assert not skipped
    with pytest.raises(EvidenceError, match="limit"):
        expand_files([tmp_path], 0)


def test_pdf_page_limit(monkeypatch):
    from evidencelab import documents

    monkeypatch.setattr(documents, "MAX_PAGES", 1)
    with pytest.raises(EvidenceError, match="page limit"):
        extract_source("r", "x", pdf_bytes(), "pdf", "x")


def test_line_numbers_preserve_leading_blank_lines():
    source = extract_source("r", "file:///notes.txt", b"\n\nActual evidence\n", "text", "Notes")
    passage = chunk_source(source)[0]
    assert passage.locator == "Document, lines 3-3"
    assert passage.text == "Actual evidence"


def test_whitespace_only_document_is_not_evidence():
    with pytest.raises(EvidenceError, match="no extractable text"):
        extract_source("r", "file:///blank.txt", b"\n  \t", "text", "Blank")
