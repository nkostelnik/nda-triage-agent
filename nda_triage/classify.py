"""Stage 4: reasoning. Claude reads the NDA; the rubric decides the colors.

One call per rubric rule (9 calls), plus one posture cross-check call:

    Document --render--> clause-tagged text (cached once, reused by all 10 calls)
        + one rule's patterns and modifiers
        --> Claude (structured output, validated by Pydantic)
        --> which patterns matched, citing clause ids and verbatim quotes
        --> Python looks up each pattern's color for the stated posture,
            applies modifiers, and takes the worst color

Design choices:

- Per rule, not per clause. Several playbook positions are about what is
  missing (no termination right, no trade-secret carve-out, missing
  exclusions). A call that sees one clause can't see an absence; a call that
  reads the whole NDA for one rule can. Each call also has one job, which
  keeps it focused. The cost is sending the NDA ten times, which prompt
  caching makes cheap: the system prompt and document are an identical
  prefix on every call, so calls 2-10 read them from cache.
- The model finds facts; it never picks a color. It is not even told the
  posture. It reports which pattern matched and where, and the color comes
  from the attorney's rubric in Python. That keeps the risk call
  deterministic, auditable, and testable without the API.
- Structured output instead of free text. Each call's response schema is
  built from that rule's own pattern ids, so the model cannot name a
  pattern that doesn't exist. The SDK validates the response against the
  Pydantic model. (Forcing a specific tool call isn't available on this
  model, so structured outputs are the supported way to force JSON.)
- Trust, but verify. Every cited clause id must exist, and every quote must
  actually appear in the cited clause. A finding that fails the check is
  marked unverified and the rule can't come back GREEN.
- Fail closed. If a rule can't be resolved (no pattern matched, a quote
  didn't verify, or the model declined), the rule is YELLOW with a
  needs_review note. It is never silently GREEN.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError, create_model

from nda_triage.ingest import normalize, parse_nda
from nda_triage.models import Document, Finding, PostureCheck, RuleResult, TriageResult, Usage
from nda_triage.rubric import Color, Posture, Rubric, Rule, load_rubric

MODEL = "claude-sonnet-5-5"
# If the model declines a request, the API retries it on a fallback model
# that Anthropic picks by refusal category. Any model that served a call is
# recorded in Usage.models.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens for claude-sonnet-5-5 (Oct 2026). Check current pricing.
PRICE = {"input": 2.00, "cache_write": 2.50, "cache_read": 0.20, "output": 10.00}

SYSTEM_PROMPT = """\
You are assisting a commercial contracts attorney who is triaging a nondisclosure agreement (NDA) against their playbook.

Your job is to find facts, not to judge risk. For one playbook rule at a time, you identify which of the rule's fact patterns the NDA matches, and where. The attorney's rubric, not you, decides what each pattern means for the deal.

The NDA is given as clauses, each wrapped in <clause id="...">. When you cite a clause, use its id exactly as given.

How to answer:
- Report every pattern the NDA matches for the rule. A clause can match more than one pattern, and a pattern can rest on more than one clause.
- Read the whole document. Terms that matter are often outside the section where you would expect them, for example a restriction inside "Miscellaneous" or "Cooperation". Judge clauses by what they do, not by their headings or labels; a clause called "mutual" may not be.
- For each match, quote the operative words that establish it, copied exactly from one cited clause as a single continuous excerpt. No ellipses, no paraphrase.
- Some patterns describe an absence (for example, "no non-solicit of any kind"). When one applies, report it with no clause ids and an empty quote.
- If nothing in the NDA addresses the rule and no pattern describes the absence, return no findings.
- Report a modifier only if its condition is met.
- Each reason is one plain-English sentence."""

RANK = {Color.GREEN: 0, Color.YELLOW: 1, Color.RED: 2}


class ClassificationError(Exception):
    """The model declined or returned nothing usable for one call."""


# ---------------------------------------------------------------------------
# Prompt pieces
# ---------------------------------------------------------------------------

def render_document(doc: Document) -> str:
    clauses = "\n\n".join(f'<clause id="{c.id}">\n{c.text}\n</clause>' for c in doc.clauses)
    return f"<nda>\n{clauses}\n</nda>"


def party_line(party: str) -> str:
    return (f'The attorney\'s client in this NDA is "{party}". In the patterns, "us" and "we" '
            f'mean {party}; "them" and "they" mean the other party.')


def rule_question(rule: Rule, party: str) -> str:
    lines = [party_line(party), "", f'<rule id="{rule.id}">', f"Name: {rule.name}",
             f"Description: {rule.description}", "Patterns:"]
    lines += [f"- {p.id}: {p.description.strip()}" for p in rule.patterns]
    if rule.modifiers:
        lines.append("Modifiers:")
        lines += [f"- {m.id}: {m.condition.strip()}" for m in rule.modifiers]
    lines += ["</rule>", "", "Which of this rule's patterns does the NDA match?"]
    return "\n".join(lines)


def posture_question(party: str) -> str:
    return "\n".join([
        party_line(party),
        "",
        "Ignore the title and labels. From the substance alone (whose information is protected, "
        "and whether the confidentiality obligations run both ways), which role does this party "
        "play?",
        f"- discloser: mainly {party}'s information is protected.",
        f"- recipient: mainly the other party's information is protected.",
        "- mutual: both parties' information is protected on substantially equal terms.",
        "- unclear: the document doesn't say enough to tell.",
    ])


# ---------------------------------------------------------------------------
# Response schemas, built per rule so only real pattern ids are allowed
# ---------------------------------------------------------------------------

def assessment_schema(rule: Rule) -> type[BaseModel]:
    pattern_id = Literal[tuple(p.id for p in rule.patterns)]
    match = create_model(
        f"{rule.id}_match",
        pattern_id=(pattern_id, ...),
        clause_ids=(list[str], Field(description="Clause ids the match rests on; empty for an absence.")),
        quote=(str, Field(description="Exact words from one cited clause; empty for an absence.")),
        reason=(str, Field(description="One plain-English sentence.")),
    )
    fields = {"findings": (list[match], ...)}
    if rule.modifiers:
        modifier = create_model(
            f"{rule.id}_modifier",
            modifier_id=(Literal[tuple(m.id for m in rule.modifiers)], ...),
            reason=(str, ...),
        )
        fields["triggered_modifiers"] = (list[modifier], ...)
    return create_model(f"{rule.id}_assessment", **fields)


class PostureAssessment(BaseModel):
    apparent_posture: Literal["discloser", "recipient", "mutual", "unclear"]
    reason: str


# ---------------------------------------------------------------------------
# The API call
# ---------------------------------------------------------------------------

def doc_block_for(doc: Document) -> dict:
    """The document as a content block, marked as the end of the cacheable prefix."""
    return {"type": "text", "text": render_document(doc), "cache_control": {"type": "ephemeral"}}


def ask(client, doc_block: dict | None, question: str, schema: type[BaseModel],
        system: str = SYSTEM_PROMPT) -> tuple[BaseModel, Usage]:
    """One structured-output call. The system prompt and the document come
    first and never change between calls, so they form the cached prefix;
    only the question after them varies. doc_block may be None for a call
    that doesn't need the document."""
    content = ([doc_block] if doc_block else []) + [{"type": "text", "text": question}]
    try:
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
    except ValidationError as e:  # the response didn't fit the schema
        raise ClassificationError(f"response failed validation: {e.error_count()} error(s)") from e
    u = response.usage
    usage = Usage(
        calls=1,
        input_tokens=u.input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        models=[response.model],
    )
    if response.stop_reason == "refusal":
        raise ClassificationError("the model declined this request")
    if response.parsed_output is None:
        raise ClassificationError(f"no structured output (stop_reason={response.stop_reason})")
    return response.parsed_output, usage


# ---------------------------------------------------------------------------
# Scoring: deterministic, no API involved
# ---------------------------------------------------------------------------

def worst(colors) -> Color:
    return max(colors, key=RANK.__getitem__, default=Color.GREEN)


def bump(color: Color) -> Color:
    return Color.YELLOW if color == Color.GREEN else Color.RED


def quote_is_in(quote: str, clause_ids: list[str], doc: Document) -> bool:
    if not quote.strip():
        return not clause_ids  # an empty quote is only fine for an absence
    haystack = " ".join(doc.clause(cid).text for cid in clause_ids if cid in {c.id for c in doc.clauses})
    return normalize(quote).lower() in normalize(haystack).lower()


def score_rule(rule: Rule, assessment, doc: Document, posture: Posture) -> RuleResult:
    patterns = {p.id: p for p in rule.patterns}
    known_ids = {c.id for c in doc.clauses}

    findings = []
    for m in assessment.findings:
        verified = all(cid in known_ids for cid in m.clause_ids) and quote_is_in(m.quote, m.clause_ids, doc)
        findings.append(Finding(
            pattern_id=m.pattern_id,
            clause_ids=m.clause_ids,
            quote=m.quote,
            reason=m.reason,
            color=patterns[m.pattern_id].colors[posture].value,
            quote_verified=verified,
        ))

    if not findings:
        return RuleResult(
            rule_id=rule.id, color=Color.YELLOW.value, findings=[],
            needs_review="No pattern matched and the rubric has no position for that. Counsel should check this rule.",
        )

    color = worst(Color(f.color) for f in findings)
    applied = []
    for hit in getattr(assessment, "triggered_modifiers", []):
        modifier = next(m for m in rule.modifiers if m.id == hit.modifier_id)
        if posture in modifier.postures:
            color = bump(color) if modifier.effect == "bump_one_color" else Color.RED
            applied.append(f"{modifier.id}: {hit.reason}")

    needs_review = None
    if not all(f.quote_verified for f in findings):
        color = worst([color, Color.YELLOW])
        needs_review = "A cited quote or clause id could not be found in the document."

    return RuleResult(rule_id=rule.id, color=color.value, findings=findings,
                      modifiers_applied=applied, needs_review=needs_review)


def total(usages: list[Usage]) -> Usage:
    return Usage(
        calls=sum(u.calls for u in usages),
        input_tokens=sum(u.input_tokens for u in usages),
        cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        models=sorted({m for u in usages for m in u.models}),
    )


def estimated_cost(u: Usage) -> float:
    return (u.input_tokens * PRICE["input"] + u.cache_write_tokens * PRICE["cache_write"]
            + u.cache_read_tokens * PRICE["cache_read"] + u.output_tokens * PRICE["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def triage(doc: Document, rubric: Rubric, posture: Posture, party: str, client=None, max_workers: int = 5) -> TriageResult:
    client = client or anthropic.Anthropic()
    doc_block = doc_block_for(doc)
    usages: list[Usage] = []

    # The first call runs alone so it writes the cache; the rule calls that
    # follow run in parallel and all read it.
    posture_check = None
    if rubric.posture_cross_check:
        try:
            pa, u = ask(client, doc_block, posture_question(party), PostureAssessment)
            usages.append(u)
            posture_check = PostureCheck(stated=posture.value, apparent=pa.apparent_posture,
                                         consistent=pa.apparent_posture == posture.value, reason=pa.reason)
        except ClassificationError as e:
            posture_check = PostureCheck(stated=posture.value, apparent="unclear", consistent=False, reason=str(e))

    def run(rule: Rule) -> RuleResult:
        try:
            assessment, u = ask(client, doc_block, rule_question(rule, party), assessment_schema(rule))
        except ClassificationError as e:
            return RuleResult(rule_id=rule.id, color=Color.YELLOW.value, findings=[],
                              needs_review=f"Analysis failed: {e}.")
        usages.append(u)
        return score_rule(rule, assessment, doc, posture)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(run, rubric.rules))

    return TriageResult(
        source_name=doc.source_name,
        sha256=doc.sha256,
        posture=posture.value,
        party=party,
        color=worst(Color(r.color) for r in results).value,
        rules=results,
        posture_check=posture_check,
        usage=total(usages),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an NDA against the playbook rubric.")
    parser.add_argument("path")
    parser.add_argument("--posture", required=True, choices=[p.value for p in Posture])
    parser.add_argument("--party", required=True, help='your client\'s name as it appears in the NDA, e.g. "Acme Analytics, Inc."')
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    result = triage(parse_nda(args.path), load_rubric(), Posture(args.posture), args.party)
    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(f"{result.source_name}: {result.color}   ({result.posture}, for {result.party})")
    print(f"{result.notice}\n")
    if result.posture_check and not result.posture_check.consistent:
        print(f"POSTURE WARNING: stated {result.posture_check.stated}, document reads as "
              f"{result.posture_check.apparent}. {result.posture_check.reason}\n")
    for r in result.rules:
        print(f"{r.color:<7} {r.rule_id}")
        for f in r.findings:
            where = ", ".join(f.clause_ids) or "(absence)"
            check = "" if f.quote_verified else "  [QUOTE NOT VERIFIED]"
            print(f"          {f.color:<7} {f.pattern_id} @ {where}{check}")
            print(f"                  {f.reason}")
        for m in r.modifiers_applied:
            print(f"          modifier: {m}")
        if r.needs_review:
            print(f"          needs review: {r.needs_review}")
    u = result.usage
    print(f"\n{u.calls} calls via {', '.join(u.models)}: input {u.input_tokens}, cache write "
          f"{u.cache_write_tokens}, cache read {u.cache_read_tokens}, output {u.output_tokens} tokens; "
          f"about ${estimated_cost(u):.3f}")


if __name__ == "__main__":
    main()
