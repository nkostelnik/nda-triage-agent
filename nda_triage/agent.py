"""Stage 5: the redline agent.

After classification, every YELLOW or RED finding goes through a bounded loop:

    1. Retrieve  the playbook fallback: exact rubric lookup first, vector
                 search (restricted to the rule) only if the rubric has none.
    2. Draft     a redline with Claude, grounded in that fallback.
    3. Check     the draft two ways:
                   a. Re-review: apply the redline to a copy of the NDA and run
                      the same rule classifier from Stage 4 on it. The issue
                      must no longer be flagged.
                   b. Fidelity: a separate call confirms the draft adds nothing
                      beyond the original clause and the fallback.
    4. Revise    once, feeding the check's issues back, if the check failed.
                 If the second draft also fails, it is returned marked
                 needs_attorney rather than retried forever.

Design choices:

- A bounded workflow, not an open-ended agent. The steps are fixed and the
  loop runs at most twice, so cost is predictable and every run can be
  audited step by step. For legal work that trade is worth more than the
  flexibility of letting the model decide what to do next.
- The self-check reuses the reviewer. "Does the fix work?" is answered by
  the same per-rule classifier and the same rubric that raised the flag,
  rather than by asking the drafter whether it did a good job.
- Grounding. The drafter is told to use the playbook fallback's language and
  to adapt only names, defined terms, and cross-references. The fidelity
  check enforces that.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from nda_triage.classify import (
    ClassificationError, ask, assessment_schema, doc_block_for, party_line, rule_question, score_rule, total, triage,
    estimated_cost,
)
from nda_triage.ingest import assemble, parse_nda
from nda_triage.models import Document, Finding, Redline, RuleResult, SelfCheck, TriageResult, Usage
from nda_triage.retrieval import Retriever, lookup_fallbacks
from nda_triage.rubric import Color, Fallback, Posture, Rubric, Rule, load_rubric

DRAFT_SYSTEM = """\
You draft redlines for a commercial contracts attorney reviewing a nondisclosure agreement (NDA). The NDA is given as clauses, each wrapped in <clause id="...">.

You will be given one issue the attorney's playbook flagged and the playbook's fallback language for it. Write the redline that fixes the issue using that fallback.

Rules:
- Ground the change in the fallback. Use its wording. Adapt only party names, defined terms, and section cross-references so it fits this NDA.
- Add no obligation, right, number, or term that appears in neither the original clause nor the fallback.
- Change only what the issue requires. Keep unrelated language in the clause as it is.
- If the fallback says to delete a provision, remove only the language that creates the issue. If nothing meaningful would be left of the clause, use the delete operation.
- If no clause was cited because the NDA lacks something, insert the fallback as a new clause and choose the clause it should follow.
- For replace, give the complete revised clause, including its number and heading. For insert, give the complete new clause. For delete, leave proposed_text empty.
- The explanation is one plain-English sentence for the attorney."""

CHECK_SYSTEM = """\
You check a drafted NDA redline for a commercial contracts attorney. You are given the original clause, the playbook fallback the redline was supposed to use, and the proposed change.

The redline is faithful only if all of these hold:
1. It carries out the fallback's substance.
2. It adds no obligation, right, number, or term that appears in neither the original clause nor the fallback. Adapting names, defined terms, and cross-references is fine.
3. Language unrelated to the issue is preserved.

List each problem you find as one plain sentence. If there are none, return an empty list."""

MAX_ATTEMPTS = 2


class Draft(BaseModel):
    operation: Literal["replace", "delete", "insert"]
    clause_id: str = Field(description="replace/delete: the clause changed. insert: the clause the new one follows.")
    proposed_text: str = Field(description="Full revised clause (replace) or new clause (insert); empty for delete.")
    explanation: str


class Fidelity(BaseModel):
    faithful: bool
    issues: list[str]


# ---------------------------------------------------------------------------
# 1. Which findings need a redline, and which fallback each one gets
# ---------------------------------------------------------------------------

def flagged_findings(result: RuleResult) -> list[Finding]:
    """The findings that made a rule YELLOW or RED. Unverified findings are
    skipped: drafting against a quote we couldn't find would compound an
    error, and the rule is already marked needs_review."""
    if result.color == Color.GREEN.value:
        return []
    verified = [f for f in result.findings if f.quote_verified]
    not_green = [f for f in verified if f.color != Color.GREEN.value]
    # A rule can turn YELLOW or RED through a modifier while its pattern is
    # GREEN (e.g. a 2-year term with no termination right).
    return not_green or (verified if result.modifiers_applied else [])


def choose_fallback(rubric: Rubric, retriever, rule: Rule, finding: Finding, posture: Posture,
                    doc: Document) -> tuple[Fallback | None, str, list[str]]:
    query = " ".join(doc.clause(c).text for c in finding.clause_ids) or f"{rule.description} {finding.reason}"
    semantic_top = [h.id for h in retriever.search(query, posture, k=3)]

    exact = lookup_fallbacks(rubric, rule.id, finding.pattern_id, posture)
    if exact:
        return exact[0], "exact", semantic_top

    in_rule = retriever.search(query, posture, rule_id=rule.id, k=1)
    if in_rule:
        fallbacks = {fb.id: fb for fb in rule.fallbacks}
        return fallbacks[in_rule[0].id], "semantic", semantic_top
    return None, "none", semantic_top


# ---------------------------------------------------------------------------
# 2. Draft
# ---------------------------------------------------------------------------

def draft_question(rule: Rule, finding: Finding, fallback: Fallback, posture: Posture, party: str,
                   previous: Draft | None, issues: list[str]) -> str:
    pattern = next(p for p in rule.patterns if p.id == finding.pattern_id)
    cited = ", ".join(finding.clause_ids) or "none (the NDA lacks this)"
    lines = [
        party_line(party),
        f"The client is the {posture.value} party in this NDA.",
        "",
        f"Issue: {rule.name}. {pattern.description.strip()}",
        f"Reviewer's note: {finding.reason}",
        f"Cited clauses: {cited}",
        "",
        f"Playbook fallback ({fallback.action}):",
        fallback.text.strip(),
    ]
    if fallback.notes:
        lines.append(f"Playbook note: {fallback.notes}")
    if previous:
        lines += ["", "Your previous draft failed review:",
                  f"operation={previous.operation}, clause_id={previous.clause_id}",
                  previous.proposed_text or "(empty)",
                  "Problems:", *[f"- {i}" for i in issues],
                  "Write a corrected redline."]
    else:
        lines += ["", "Write the redline."]
    return "\n".join(lines)


def structural_problems(draft: Draft, finding: Finding, doc: Document) -> list[str]:
    ids = {c.id for c in doc.clauses}
    if draft.clause_id not in ids:
        return [f"clause_id {draft.clause_id!r} is not a clause in this NDA."]
    if draft.operation in ("replace", "delete") and finding.clause_ids and draft.clause_id not in finding.clause_ids:
        return [f"{draft.operation} must target one of the cited clauses: {', '.join(finding.clause_ids)}."]
    if draft.operation != "delete" and not draft.proposed_text.strip():
        return [f"{draft.operation} needs proposed_text."]
    return []


def apply_draft(doc: Document, draft: Draft) -> Document:
    """A copy of the document with the redline applied."""
    parts = []
    for c in doc.clauses:
        if c.id == draft.clause_id and draft.operation == "replace":
            parts.append((c.id, c.number, c.heading, draft.proposed_text))
        elif c.id == draft.clause_id and draft.operation == "delete":
            continue
        else:
            parts.append((c.id, c.number, c.heading, c.text))
            if c.id == draft.clause_id and draft.operation == "insert":
                parts.append(("inserted", None, None, draft.proposed_text))
    return assemble(doc.source_name, doc.sha256, parts)


# ---------------------------------------------------------------------------
# 3. Check
# ---------------------------------------------------------------------------

def recheck(client, rubric: Rubric, rule: Rule, finding: Finding, draft: Draft, doc: Document, posture: Posture,
            party: str) -> tuple[RuleResult, bool, Usage]:
    """Run the Stage 4 classifier for this rule on the revised document."""
    revised = apply_draft(doc, draft)
    assessment, usage = ask(client, doc_block_for(revised), rule_question(rule, party), assessment_schema(rubric))
    after = score_rule(rule, assessment, revised, posture)
    target = "inserted" if draft.operation == "insert" else draft.clause_id
    still_flagged = [
        f for f in after.findings
        if f.color != Color.GREEN.value and (f.pattern_id == finding.pattern_id or target in f.clause_ids)
    ]
    resolved = after.color == Color.GREEN.value or (not still_flagged and not after.modifiers_applied)
    return after, resolved, usage


def fidelity_question(draft: Draft, fallback: Fallback, doc: Document) -> str:
    original = doc.clause(draft.clause_id).text if draft.operation != "insert" else "(none: this is a new clause)"
    return "\n".join([
        "Original clause:", original, "",
        f"Playbook fallback ({fallback.action}):", fallback.text.strip(), "",
        f"Proposed change ({draft.operation}):", draft.proposed_text or "(delete the clause)",
    ])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def redline(client, rubric: Rubric, retriever, rule: Rule, finding: Finding, doc: Document,
            posture: Posture, party: str) -> tuple[Redline, list[Usage]]:
    usages: list[Usage] = []
    fallback, method, semantic_top = choose_fallback(rubric, retriever, rule, finding, posture, doc)
    base = dict(rule_id=rule.id, pattern_id=finding.pattern_id, retrieval=method, semantic_top=semantic_top)
    if fallback is None:
        return Redline(**base, status="not_drafted", note="The playbook has no fallback for this issue."), usages

    doc_block = doc_block_for(doc)
    draft, issues = None, []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resolved, faithful, color_after = False, False, "unknown"
        try:
            question = draft_question(rule, finding, fallback, posture, party, draft, issues)
            draft, u = ask(client, doc_block, question, Draft, system=DRAFT_SYSTEM)
            usages.append(u)
            issues = structural_problems(draft, finding, doc)
            if not issues:
                after, resolved, u = recheck(client, rubric, rule, finding, draft, doc, posture, party)
                usages.append(u)
                color_after = after.color
                if not resolved:
                    issues.append(f"Reviewing the revised NDA still flags this rule ({after.color}): " + "; ".join(
                        f"{f.pattern_id}: {f.reason}" for f in after.findings if f.color != Color.GREEN.value))

                fidelity, u = ask(client, None, fidelity_question(draft, fallback, doc), Fidelity, system=CHECK_SYSTEM)
                usages.append(u)
                faithful = fidelity.faithful
                if not faithful:
                    issues += fidelity.issues or ["The fidelity check found the draft unfaithful to the fallback."]
        except ClassificationError as e:
            return Redline(**base, fallback_id=fallback.id, status="needs_attorney",
                           note=f"Drafting failed: {e}."), usages
        if not issues:
            break

    replaced = draft.operation != "insert" and draft.clause_id in {c.id for c in doc.clauses}
    return Redline(
        **base,
        fallback_id=fallback.id,
        operation=draft.operation,
        clause_id=draft.clause_id,
        original_text=doc.clause(draft.clause_id).text if replaced else "",
        proposed_text=draft.proposed_text,
        explanation=draft.explanation,
        check=SelfCheck(resolved=resolved, rule_color_after=color_after, faithful=faithful,
                        issues=issues, attempts=attempt),
        status="needs_attorney" if issues else "passed",
    ), usages


def review(doc: Document, rubric: Rubric, posture: Posture, party: str, client=None, retriever=None,
           max_workers: int = 4) -> TriageResult:
    """Classify the NDA (Stage 4), then draft and check a redline for every flag."""
    client = client or anthropic.Anthropic()
    retriever = retriever or Retriever(rubric)
    result = triage(doc, rubric, posture, party, client=client)

    # One redline per (rule, fallback, clause). Two findings about the same
    # clause that the same fallback fixes (e.g. a non-solicit that is both
    # too broad and too long) get one redline, not two identical ones. The
    # re-review still checks every flag on that clause.
    jobs, seen = [], set()
    for rule_result in result.rules:
        rule = rubric.rule(rule_result.rule_id)
        for finding in flagged_findings(rule_result):
            exact = lookup_fallbacks(rubric, rule.id, finding.pattern_id, posture)
            key = (rule.id, exact[0].id if exact else finding.pattern_id,
                   finding.clause_ids[0] if finding.clause_ids else None)
            if key not in seen:
                seen.add(key)
                jobs.append((rule, finding))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outcomes = list(pool.map(lambda job: redline(client, rubric, retriever, *job, doc, posture, party), jobs))

    result.redlines = [r for r, _ in outcomes]
    result.usage = total([result.usage, *[u for _, us in outcomes for u in us]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an NDA and draft redlines for every flag.")
    parser.add_argument("path")
    parser.add_argument("--posture", required=True, choices=[p.value for p in Posture])
    parser.add_argument("--party", required=True, help="your client's name as it appears in the NDA")
    args = parser.parse_args()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    result = review(parse_nda(args.path), load_rubric(), Posture(args.posture), args.party)
    print(f"{result.source_name}: {result.color}   ({result.posture}, for {result.party})")
    print(f"{result.notice}\n")
    for r in result.rules:
        print(f"{r.color:<7} {r.rule_id}" + (f"   needs review: {r.needs_review}" if r.needs_review else ""))
    for rl in result.redlines:
        print(f"\n--- {rl.rule_id} / {rl.pattern_id}: {rl.status.upper()} "
              f"(fallback {rl.fallback_id}, {rl.retrieval} retrieval)")
        if rl.operation:
            print(f"{rl.operation} {rl.clause_id}: {rl.explanation}")
            if rl.original_text:
                print(f"  BEFORE: {rl.original_text}")
            if rl.proposed_text:
                print(f"  AFTER:  {rl.proposed_text}")
        if rl.check and rl.check.issues:
            print("  check issues: " + " | ".join(rl.check.issues))
        if rl.note:
            print(f"  note: {rl.note}")
    u = result.usage
    print(f"\n{u.calls} calls via {', '.join(u.models)}; about ${estimated_cost(u):.3f}")


if __name__ == "__main__":
    main()
