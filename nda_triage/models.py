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


class RetrievedText(BaseModel):
    """One retrieval hit: a playbook fallback or a reference-corpus clause."""

    source: str = Field(description="'playbook' or 'corpus'.")
    id: str = Field(description="Fallback id, or 'file:clause-id' for corpus chunks.")
    rule_id: str | None = None
    action: str | None = Field(None, description="'replace' or 'strike' (playbook fallbacks only).")
    text: str
    distance: float = Field(description="Cosine distance to the query; lower is closer.")


class Finding(BaseModel):
    """One fact the model found for a rule, with the color the rubric assigns it."""

    pattern_id: str
    clause_ids: list[str] = Field(description="Clauses the finding rests on; empty if the finding is an absence.")
    quote: str = Field(description="Verbatim excerpt from the cited clause(s); empty for an absence.")
    reason: str = Field(description="One-line reason, from the model.")
    color: str = Field(description="Looked up in the rubric for the stated posture, not chosen by the model.")
    quote_verified: bool = Field(description="True if the quote really appears in the cited clauses.")


class RuleResult(BaseModel):
    rule_id: str
    color: str
    findings: list[Finding]
    modifiers_applied: list[str] = []
    needs_review: str | None = Field(None, description="Why the rule could not be resolved automatically, if it couldn't.")


class PostureCheck(BaseModel):
    stated: str
    apparent: str = Field(description="What the document's structure suggests: discloser, recipient, mutual, or unclear.")
    consistent: bool
    reason: str


class Usage(BaseModel):
    """Token accounting across all calls for one NDA."""

    calls: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    models: list[str] = Field([], description="Every model that served a call (a fallback model would appear here).")


class SelfCheck(BaseModel):
    """The agent's review of its own draft before returning it."""

    resolved: bool = Field(description="Re-reviewing the revised document no longer flags this issue.")
    rule_color_after: str = Field(description="The rule's color when the revised document is reviewed again.")
    faithful: bool = Field(description="The draft adds nothing beyond the original clause and the playbook fallback.")
    issues: list[str] = []
    attempts: int = Field(description="Drafts written: 1, or 2 if the first failed the check.")


class Redline(BaseModel):
    rule_id: str
    pattern_id: str
    fallback_id: str | None = None
    retrieval: str = Field(description="'exact' (rubric lookup), 'semantic' (vector search), or 'none'.")
    semantic_top: list[str] = Field([], description="Vector search's top fallback ids, kept for the evals.")
    operation: str | None = Field(None, description="'replace', 'delete', or 'insert'.")
    clause_id: str | None = Field(None, description="Clause replaced or deleted, or the clause an insert follows.")
    original_text: str = ""
    proposed_text: str = ""
    explanation: str = ""
    check: SelfCheck | None = None
    status: str = Field(description="'passed', 'needs_attorney', or 'not_drafted'.")
    note: str | None = None


class TriageResult(BaseModel):
    source_name: str
    sha256: str
    posture: str
    party: str
    color: str
    rules: list[RuleResult]
    posture_check: PostureCheck | None
    usage: Usage
    redlines: list[Redline] = []
    notice: str = "Requires attorney review. Not legal advice."


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
