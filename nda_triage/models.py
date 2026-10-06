"""Data shapes passed between pipeline stages.

Pydantic models give each stage a clear contract and validate data at the
boundary, so a malformed value fails loudly where it enters instead of deep
inside a later stage. Later stages add Flag, Redline, and Verdict here.
"""

from pydantic import BaseModel, Field


class Clause(BaseModel):
    """One section of the NDA, the unit that every flag points back to."""

    id: str = Field(description="Stable ID used to trace flags to source text, e.g. 'sec-4.2'.")
    number: str | None = Field(None, description="Section number as shown in the document, e.g. '4.2'.")
    heading: str | None = Field(None, description="Section heading, e.g. 'Confidential Information'.")
    text: str = Field(description="Normalized clause text, including its number and heading.")
    start: int = Field(description="Character offset where the clause begins in Document.text.")
    end: int = Field(description="Character offset where the clause ends in Document.text.")


class Document(BaseModel):
    """A parsed NDA. Invariant: text[clause.start:clause.end] == clause.text."""

    source_name: str
    sha256: str = Field(description="Hash of the original file bytes, tying results to an exact input.")
    text: str = Field(description="Full normalized document text.")
    clauses: list[Clause]

    def clause(self, clause_id: str) -> Clause:
        for c in self.clauses:
            if c.id == clause_id:
                return c
        raise KeyError(clause_id)
