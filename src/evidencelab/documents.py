"""Extraction from explicit local documents and fetched web resources."""

import io
import os
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError
from trafilatura import bare_extraction

from evidencelab.evidence import (
    MAX_BYTES,
    MAX_PAGES,
    EvidenceError,
    Segment,
    Source,
    SourceKind,
    digest,
)
from evidencelab.fetching import fetch_url, normalize_url

SUPPORTED = {".md", ".txt", ".pdf"}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def expand_files(paths: list[Path], limit: int) -> tuple[list[Path], list[dict[str, str]]]:
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.is_symlink() or path.is_junction():
            skipped.append(
                {"input": str(path), "reason": "Symbolic links and junctions are skipped."}
            )
        elif not path.is_file() or path.suffix.lower() not in SUPPORTED:
            skipped.append(
                {"input": str(path), "reason": "Not a supported Markdown, text, or PDF file."}
            )
        else:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
                if len(files) > limit:
                    raise EvidenceError(f"Input set exceeds the {limit}-source limit.")

    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_symlink() or path.is_junction() or not path.is_dir():
            add(path)
            continue
        for root, directories, names in os.walk(path, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if not name.startswith(".")
                and not (Path(root) / name).is_symlink()
                and not (Path(root) / name).is_junction()
            )
            for name in sorted(names):
                if not name.startswith(".") and Path(name).suffix.lower() in SUPPORTED:
                    add(Path(root) / name)
    return files, skipped


def extract_source(
    run_id: str, location: str, content: bytes, kind: SourceKind, title: str
) -> Source:
    if len(content) > MAX_BYTES:
        raise EvidenceError("Document exceeds the 20 MB input limit.")
    warnings: list[str] = []
    published_at = None
    try:
        if kind == "pdf":
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise EvidenceError("Encrypted PDFs are unsupported.")
            if len(reader.pages) > MAX_PAGES:
                raise EvidenceError("PDF exceeds the 300-page limit.")
            segments = []
            for number, page in enumerate(reader.pages, 1):
                text = normalize_text(page.extract_text() or "")
                if text.strip():
                    segments.append(Segment(text=text, locator=f"PDF page {number}", page=number))
                else:
                    warnings.append(
                        f"PDF page {number} has no extractable text; OCR is unavailable."
                    )
                if sum(len(s.text) for s in segments) > MAX_BYTES:
                    raise EvidenceError("Extracted text exceeds the 20 MB limit.")
        elif kind == "html":
            document = bare_extraction(
                content,
                url=location,
                with_metadata=True,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if document is None or isinstance(document, dict):
                raise EvidenceError("No readable article text was found on this page.")
            title = document.title or title
            published_at = document.date
            segments = [
                Segment(text=normalize_text(document.text or ""), locator="Extracted article")
            ]
        else:
            text = normalize_text(content.decode("utf-8-sig"))
            if "\x00" in text:
                raise EvidenceError("Binary content is not a supported text document.")
            segments = [Segment(text=text, locator="Document")]
            if kind == "markdown":
                heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if heading:
                    title = heading[1].strip()
    except (UnicodeError, PyPdfError, ValueError, RecursionError):
        raise EvidenceError(
            "Document could not be parsed; text must be UTF-8 and PDFs must be valid."
        ) from None
    if not any(segment.text.strip() for segment in segments):
        raise EvidenceError("Document has no extractable text; scanned PDFs require OCR.")
    if sum(len(segment.text) for segment in segments) > MAX_BYTES:
        raise EvidenceError("Extracted text exceeds the 20 MB limit.")
    normalized = " ".join(" ".join(s.text.split()) for s in segments)
    content_hash = digest(normalized)
    return Source(
        source_id=digest(run_id, location, content_hash),
        run_id=run_id,
        location=location,
        title=title,
        kind=kind,
        content_hash=content_hash,
        published_at=published_at,
        segments=segments,
        warnings=warnings,
    )


def read_file(run_id: str, path: Path) -> Source:
    if path.is_symlink() or path.is_junction() or path.suffix.lower() not in SUPPORTED:
        raise EvidenceError("Use a regular Markdown, text, or PDF file.")
    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_BYTES + 1)
    except OSError:
        raise EvidenceError("Cannot read document; check the path and permissions.") from None
    kinds: dict[str, SourceKind] = {".md": "markdown", ".txt": "text", ".pdf": "pdf"}
    return extract_source(
        run_id, path.resolve().as_uri(), content, kinds[path.suffix.lower()], path.name
    )


def read_url(run_id: str, url: str) -> Source:
    requested = normalize_url(url)
    document = fetch_url(requested)
    kind: SourceKind
    if document.content.startswith(b"%PDF-") or document.content_type == "application/pdf":
        kind = "pdf"
    elif document.content_type in {"text/plain", "text/markdown"}:
        kind = "markdown" if document.content_type == "text/markdown" else "text"
    elif document.content_type in {"text/html", "application/xhtml+xml", ""}:
        kind = "html"
    else:
        raise EvidenceError("URL returned an unsupported document type.")
    source = extract_source(run_id, document.url, document.content, kind, document.url)
    if requested != document.url:
        source.aliases.append(requested)
    return source
