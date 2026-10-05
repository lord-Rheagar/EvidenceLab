import json

import pytest
from pydantic import ValidationError

from evidencelab.citation_references import CitationReferences, source_excerpts
from evidencelab.evidence import EvidenceError, Passage
from evidencelab.research import evidence_context, validate_claims
from evidencelab.research_records import Citation, Claim, EvidenceReview, ResearchReport


@pytest.fixture
def evidence():
    return {
        letter * 64: Passage(
            passage_id=letter * 64,
            run_id="run",
            source_id=letter,
            text=text,
            title="Technical documentation",
            location=f"https://docs.example/{letter}",
            locator="paragraph 1",
            segment=0,
            start=0,
            end=len(text),
            retrieved_at="2026-10-04T00:00:00Z",
        )
        for letter, text in [
            ("a", "Keyword search supports exact identifiers."),
            ("b", "Semantic search finds related concepts."),
        ]
    }


def review_data(reference="e1"):
    return {
        "findings": [{"text": "A supported claim.", "citations": [{"excerpt_id": reference}]}],
        "gaps": [],
        "follow_up_queries": [],
    }


def canonical_review(identifier, quote):
    return EvidenceReview(
        findings=[Claim(text="Claim", citations=[Citation(passage_id=identifier, quote=quote)])],
        gaps=[],
        follow_up_queries=[],
    )


def test_choices_are_bounded_deterministic_and_resolve_to_exact_citations(evidence):
    references = CitationReferences(evidence)
    assert (
        CitationReferences(dict(reversed(list(evidence.items())))).excerpts == references.excerpts
    )
    schema = references.schema(EvidenceReview)
    assert schema.model_json_schema()["$defs"]["ExcerptReference"]["enum"] == ["e1", "e2"]
    response = schema.model_validate(review_data())
    resolved = references.resolve(response, EvidenceReview)
    citation = resolved.findings[0].citations[0]
    assert citation.passage_id == "a" * 64
    assert citation.quote == evidence["a" * 64].text
    assert response.model_dump(mode="json")["findings"][0]["citations"][0] == {"excerpt_id": "e1"}
    validate_claims(resolved.findings, evidence)
    assert EvidenceReview.model_validate_json(resolved.model_dump_json()) == resolved


@pytest.mark.parametrize("reference", ["e0", "e3", "E1", "e01", "p1", "a" * 64, "a" * 63])
def test_unknown_or_malformed_choices_are_rejected(evidence, reference):
    with pytest.raises(ValidationError):
        CitationReferences(evidence).schema(EvidenceReview).model_validate(review_data(reference))


def test_context_and_saved_review_use_same_choices_without_mutating_storage(evidence):
    saved = canonical_review("b" * 64, evidence["b" * 64].text)
    references = CitationReferences(evidence, review=saved)
    encoded = json.loads(references.review_context(saved))
    assert encoded["findings"][0]["citations"][0] == {"excerpt_id": "e2"}
    context = json.loads(evidence_context(evidence, references))
    assert context[1]["excerpts"] == [
        {"excerpt_id": "e2", "text": saved.findings[0].citations[0].quote}
    ]
    assert saved.findings[0].citations[0].passage_id == "b" * 64
    # Smoke tests continue to receive canonical IDs and unmodified text.
    assert json.loads(evidence_context(evidence))[0]["passage_id"] == "a" * 64
    assert json.loads(evidence_context(evidence))[0]["text"] == evidence["a" * 64].text


@pytest.mark.parametrize(
    "extra",
    [
        {"quote": "An invented or paraphrased quotation."},
        {"passage_id": "b" * 64},
        {"quote": "Semantic search finds related concepts."},
    ],
)
def test_model_cannot_override_excerpt_text_or_pair_it_with_another_passage(evidence, extra):
    data = review_data()
    data["findings"][0]["citations"][0].update(extra)
    with pytest.raises(ValidationError):
        CitationReferences(evidence).schema(EvidenceReview).model_validate(data)


def test_all_report_claim_sections_are_resolved(evidence):
    references = CitationReferences(evidence)
    data = {
        "title": "Research report",
        "summary": review_data()["findings"],
        "findings": review_data("e2")["findings"],
        "tradeoffs": review_data()["findings"],
        "limitations": ["Limited evidence."],
    }
    response = references.schema(ResearchReport).model_validate(data)
    report = references.resolve(response, ResearchReport)
    claims = [*report.summary, *report.findings, *report.tradeoffs]
    assert [c.citations[0].passage_id for c in claims] == ["a" * 64, "b" * 64, "a" * 64]
    validate_claims(claims, evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("citations", []),
        ("citations", [{"excerpt_id": "e1"}] * 5),
        ("text", ""),
        ("text", "x" * 2001),
    ],
)
def test_original_claim_bounds_remain_enforced(evidence, field, value):
    data = review_data()
    data["findings"][0][field] = value
    with pytest.raises(ValidationError):
        CitationReferences(evidence).schema(EvidenceReview).model_validate(data)


def test_legacy_review_valid_substring_remains_selectable_after_resume(evidence):
    saved = canonical_review("a" * 64, "supports exact identifiers")
    references = CitationReferences(evidence, review=saved)
    encoded = references.review_context(saved)
    response = references.schema(EvidenceReview).model_validate_json(encoded)
    assert references.resolve(response, EvidenceReview) == saved
    assert {
        "excerpt_id": "e3",
        "text": "supports exact identifiers",
    } in references.passage_excerpts("a" * 64)


@pytest.mark.parametrize(
    "identifier,quote",
    [
        ("c" * 64, "Long quotation."),
        ("a" * 64, "A paraphrase that never occurred."),
        ("a" * 64, "Semantic search finds related concepts."),
        ("a" * 64, " " * 12),
    ],
)
def test_invalid_legacy_review_cannot_create_a_new_excerpt(evidence, identifier, quote):
    with pytest.raises(EvidenceError, match="do not match"):
        CitationReferences(evidence, review=canonical_review(identifier, quote))


@pytest.mark.parametrize(
    "text",
    [
        "   Keyword search\r\n  preserves\tidentifiers and  whitespace.   ",
        "Short. " + "A very long sentence with repeated words " * 100,
        "x" * 4000,
        "Alpha supports beta.\n\nGamma supports delta. " * 100,
        " Unicode punctuation: \u201ccitations\u201d retain caf\u00e9 and \u2014 text. " * 20,
    ],
)
def test_excerpt_slices_preserve_source_bytes_and_are_bounded(text):
    excerpts = source_excerpts(text)
    assert excerpts
    assert all(12 <= len(quote) <= 251 and quote in text for quote in excerpts)


def test_context_never_offers_text_outside_the_validated_4000_character_window(evidence):
    evidence["a" * 64].text = "x" * 4000 + "DO_NOT_INCLUDE_THIS_TAIL"
    references = CitationReferences(evidence)
    assert all(c.quote in evidence[c.passage_id].text[:4000] for c in references.excerpts.values())
    assert "DO_NOT_INCLUDE_THIS_TAIL" not in evidence_context(evidence, references)


def test_no_citable_text_fails_before_a_model_request(evidence):
    for passage in evidence.values():
        passage.text = "tiny"
    with pytest.raises(EvidenceError, match="minimum citation length"):
        CitationReferences(evidence)


def test_persisted_contract_still_rejects_refs_and_fabricated_quotes(evidence):
    CitationReferences(evidence).schema(EvidenceReview)
    with pytest.raises(ValidationError):
        EvidenceReview.model_validate(review_data())
    review = canonical_review("a" * 64, "A fabricated quotation.")
    with pytest.raises(EvidenceError, match="not an exact substring"):
        validate_claims(review.findings, evidence)
