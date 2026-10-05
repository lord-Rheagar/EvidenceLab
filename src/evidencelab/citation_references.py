"""Request-local excerpt choices; persisted citations contain exact snapshot text."""

import re
from copy import deepcopy
from enum import Enum
from types import GenericAlias
from typing import Any

from pydantic import BaseModel, Field, create_model

from evidencelab.evidence import EvidenceError, Passage, Record
from evidencelab.research_records import Citation, Claim, EvidenceReview, ResearchReport


def source_excerpts(text: str) -> list[str]:
    """Slice bounded excerpts without rewriting whitespace, punctuation, or spelling."""
    excerpts = []
    start = 0
    while start < len(text):
        end = min(start + 240, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"(?<=[.!?])\s+|\n+", text[start:end]))
            boundaries = [match for match in boundaries if match.end() >= 12]
            if boundaries:
                end = start + boundaries[-1].end()
            else:
                spaces = list(re.finditer(r"\s+", text[start:end]))
                if spaces and spaces[-1].start() >= 12:
                    end = start + spaces[-1].end()
        if len(text) - end < 12:
            end = len(text)
        quote = text[start:end].strip()
        if len(quote) >= 12:
            excerpts.append(quote)
        start = end
    return excerpts


class CitationReferences:
    def __init__(self, evidence: dict[str, Passage], review: EvidenceReview | None = None) -> None:
        if not evidence or any(key != passage.passage_id for key, passage in evidence.items()):
            raise EvidenceError("Citation references require matching evidence snapshots.")
        self.to_id = {f"p{n}": identifier for n, identifier in enumerate(sorted(evidence), 1)}
        self.to_ref = {identifier: ref for ref, identifier in self.to_id.items()}
        self.excerpts: dict[str, Citation] = {}
        self._references: dict[tuple[str, str], str] = {}
        for identifier in self.to_id.values():
            for quote in source_excerpts(evidence[identifier].text[:4000]):
                self._add(Citation(passage_id=identifier, quote=quote))
        # Older checkpoints may quote a different valid span. Keep that exact span selectable.
        if review is not None:
            for claim in review.findings:
                for citation in claim.citations:
                    passage = evidence.get(citation.passage_id)
                    if (
                        passage is None
                        or not citation.quote.strip()
                        or citation.quote not in passage.text[:4000]
                    ):
                        raise EvidenceError(
                            "Saved review citations do not match the supplied evidence."
                        )
                    self._add(citation)
        if not self.excerpts:
            raise EvidenceError("No source excerpts meet the minimum citation length.")

    def _add(self, citation: Citation) -> None:
        key = (citation.passage_id, citation.quote)
        if key not in self._references:
            reference = f"e{len(self.excerpts) + 1}"
            self.excerpts[reference] = citation.model_copy()
            self._references[key] = reference

    def passage_excerpts(self, identifier: str) -> list[dict[str, str]]:
        return [
            {"excerpt_id": ref, "text": citation.quote}
            for ref, citation in self.excerpts.items()
            if citation.passage_id == identifier
        ]

    def schema(self, record: type[EvidenceReview] | type[ResearchReport]) -> type[BaseModel]:
        # Members depend on this request's evidence, so they cannot be inferred statically.
        choices = Enum("ExcerptReference", {ref: ref for ref in self.excerpts}, type=str)  # type: ignore[misc]
        citation = create_model(
            "CitationChoice",
            __base__=Record,
            excerpt_id=(
                choices,
                Field(
                    description=(
                        "Select an excerpt_id that directly supports the claim. "
                        "Do not write a quote."
                    )
                ),
            ),
        )
        claim = create_model(
            "ClaimChoice",
            __base__=Claim,
            citations=(GenericAlias(list, citation), deepcopy(Claim.model_fields["citations"])),
        )
        fields: dict[str, Any] = {
            name: (GenericAlias(list, claim), deepcopy(record.model_fields[name]))
            for name in self._claim_fields(record)
        }
        return create_model(record.__name__, __base__=record, **fields)

    @staticmethod
    def _claim_fields(record: type[EvidenceReview] | type[ResearchReport]) -> tuple[str, ...]:
        return ("findings",) if record is EvidenceReview else ("summary", "findings", "tradeoffs")

    def resolve[T: (EvidenceReview, ResearchReport)](
        self, response: BaseModel, record: type[T]
    ) -> T:
        data = response.model_dump(mode="json")
        for field in self._claim_fields(record):
            for claim in data[field]:
                resolved = []
                for choice in claim["citations"]:
                    citation = self.excerpts.get(choice.get("excerpt_id"))
                    if citation is None or set(choice) != {"excerpt_id"}:
                        raise EvidenceError("Citation reference is not a supplied source excerpt.")
                    resolved.append(citation.model_dump(mode="json"))
                claim["citations"] = resolved
        return record.model_validate(data)

    def review_context(self, review: EvidenceReview) -> str:
        data = review.model_dump(mode="json")
        for claim in data["findings"]:
            choices = []
            for citation in claim["citations"]:
                reference = self._references.get((citation["passage_id"], citation["quote"]))
                if reference is None:
                    raise EvidenceError(
                        "Saved review citations do not match the supplied evidence."
                    )
                choices.append({"excerpt_id": reference})
            claim["citations"] = choices
        # Validate against the same request-local choices before including the review.
        return self.schema(EvidenceReview).model_validate(data).model_dump_json()
