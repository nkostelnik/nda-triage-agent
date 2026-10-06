"""Agent-loop tests with a scripted fake client: the reviewer answers from the
document it is shown, so a redline that really removes the non-solicit
passes re-review and one that doesn't is caught."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nda_triage.agent import CHECK_SYSTEM, DRAFT_SYSTEM, Draft, apply_draft, flagged_findings, review
from nda_triage.classify import SYSTEM_PROMPT
from nda_triage.ingest import parse_nda
from nda_triage.models import Finding, RetrievedText, RuleResult
from nda_triage.rubric import Posture, load_rubric

from test_classify import SAMPLE_ANSWERS

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_nda.txt"
PARTY = "Acme Analytics, Inc."

GOOD_DRAFT = {"operation": "replace", "clause_id": "sec-8.3",
              "proposed_text": "8.3 Cooperation. The parties will cooperate in good faith.",
              "explanation": "Removes the employee non-solicit; keeps the cooperation covenant."}
BAD_DRAFT = {"operation": "replace", "clause_id": "sec-8.3",
             "proposed_text": "8.3 Cooperation. Neither party will solicit any employee of the other party.",
             "explanation": "Narrows the clause."}
FAITHFUL = {"faithful": True, "issues": []}


class ScriptedClient:
    def __init__(self, drafts, checks=None):
        self.drafts, self.checks = list(drafts), list(checks or [FAITHFUL] * 4)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self.parse))

    def parse(self, **kw):
        system, content = kw["system"], kw["messages"][0]["content"]
        self.calls.append(system)
        if system == DRAFT_SYSTEM:
            data = self.drafts.pop(0)
        elif system == CHECK_SYSTEM:
            data = self.checks.pop(0)
        elif "<rule id=" not in content[-1]["text"]:
            data = {"apparent_posture": "mutual", "reason": "Both protected."}
        else:
            rule_id = content[-1]["text"].split('<rule id="')[1].split('"')[0]
            data = SAMPLE_ANSWERS[rule_id]
            if rule_id == "non_solicit" and "solicit" not in content[0]["text"]:
                data = {"findings": [{"pattern_id": "no_non_solicit", "clause_ids": [], "quote": "", "reason": "None."}]}
            elif rule_id == "non_solicit" and "for employment" not in content[0]["text"]:
                data = {"findings": [{"pattern_id": "broad_employee_non_solicit", "clause_ids": ["sec-8.3"],
                                      "quote": "Neither party will solicit any employee", "reason": "Still there."}]}
        usage = SimpleNamespace(input_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                                output_tokens=5)
        return SimpleNamespace(parsed_output=kw["output_format"].model_validate(data), stop_reason="end_turn",
                               usage=usage, model="claude-sonnet-5-5")


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = hits or []

    def search(self, query, posture, rule_id=None, k=3):
        return [h for h in self.hits if rule_id is None or h.rule_id == rule_id][:k]


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


@pytest.fixture(scope="module")
def doc():
    return parse_nda(SAMPLE)


def run(doc, rubric, client, retriever=None, rubric_override=None):
    return review(doc, rubric_override or rubric, Posture.mutual, PARTY, client=client,
                  retriever=retriever or FakeRetriever())


def test_flag_gets_a_grounded_redline_that_passes(doc, rubric):
    result = run(doc, rubric, ScriptedClient([GOOD_DRAFT]))
    assert len(result.redlines) == 1
    rl = result.redlines[0]
    assert (rl.rule_id, rl.fallback_id, rl.retrieval) == ("non_solicit", "non_solicit_strike", "exact")
    assert rl.status == "passed"
    assert rl.check.resolved and rl.check.faithful and rl.check.attempts == 1
    assert rl.check.rule_color_after == "GREEN"
    assert "solicit for employment" in rl.original_text


def test_failed_recheck_triggers_one_revision(doc, rubric):
    result = run(doc, rubric, ScriptedClient([BAD_DRAFT, GOOD_DRAFT]))
    rl = result.redlines[0]
    assert rl.status == "passed" and rl.check.attempts == 2
    assert rl.proposed_text == GOOD_DRAFT["proposed_text"]


def test_two_failures_go_to_the_attorney(doc, rubric):
    result = run(doc, rubric, ScriptedClient([BAD_DRAFT, BAD_DRAFT]))
    rl = result.redlines[0]
    assert rl.status == "needs_attorney" and rl.check.attempts == 2
    assert not rl.check.resolved
    assert "still flags" in rl.check.issues[0]


def test_unfaithful_draft_is_caught(doc, rubric):
    unfaithful = {"faithful": False, "issues": ["Adds a 30-day cure period not in the fallback."]}
    result = run(doc, rubric, ScriptedClient([GOOD_DRAFT, GOOD_DRAFT], checks=[unfaithful, unfaithful]))
    rl = result.redlines[0]
    assert rl.status == "needs_attorney"
    assert rl.check.resolved and not rl.check.faithful
    assert "cure period" in rl.check.issues[0]


def test_draft_targeting_the_wrong_clause_is_revised(doc, rubric):
    wrong = dict(GOOD_DRAFT, clause_id="sec-3")
    result = run(doc, rubric, ScriptedClient([wrong, GOOD_DRAFT]))
    assert result.redlines[0].status == "passed" and result.redlines[0].check.attempts == 2


def test_semantic_retrieval_used_when_rubric_has_no_exact_fallback(doc, rubric):
    edited = rubric.model_copy(deep=True)
    edited.rule("non_solicit").fallbacks[0].applies_to = ["broad_employee_non_solicit"]
    hit = RetrievedText(source="playbook", id="non_solicit_strike", rule_id="non_solicit", action="strike",
                        text="Delete...", distance=0.5)
    result = run(doc, rubric, ScriptedClient([GOOD_DRAFT]), FakeRetriever([hit]), rubric_override=edited)
    assert result.redlines[0].retrieval == "semantic"


def test_no_fallback_means_not_drafted(doc, rubric):
    edited = rubric.model_copy(deep=True)
    edited.rule("non_solicit").fallbacks = []
    result = run(doc, rubric, ScriptedClient([]), rubric_override=edited)
    assert result.redlines[0].status == "not_drafted"


def test_usage_includes_agent_calls(doc, rubric):
    client = ScriptedClient([GOOD_DRAFT])
    result = run(doc, rubric, client)
    assert result.usage.calls == len(client.calls) == 10 + 3  # classify + draft, re-review, fidelity


# ---- pure helpers ------------------------------------------------------------

def finding(color, verified=True, pattern="p"):
    return Finding(pattern_id=pattern, clause_ids=["sec-1"], quote="x", reason="r", color=color,
                   quote_verified=verified)


def test_flagged_findings_rules():
    assert flagged_findings(RuleResult(rule_id="r", color="GREEN", findings=[finding("GREEN")])) == []
    mixed = RuleResult(rule_id="r", color="RED", findings=[finding("GREEN"), finding("RED")])
    assert [f.color for f in flagged_findings(mixed)] == ["RED"]
    via_modifier = RuleResult(rule_id="r", color="YELLOW", findings=[finding("GREEN")], modifiers_applied=["m"])
    assert len(flagged_findings(via_modifier)) == 1
    unverified = RuleResult(rule_id="r", color="YELLOW", findings=[finding("YELLOW", verified=False)])
    assert flagged_findings(unverified) == []


@pytest.mark.parametrize("operation, expect_ids", [
    ("replace", None),
    ("delete", lambda ids: "sec-8.3" not in ids),
    ("insert", lambda ids: ids[ids.index("sec-8.3") + 1] == "inserted"),
])
def test_apply_draft_keeps_offsets_consistent(doc, operation, expect_ids):
    draft = Draft(operation=operation, clause_id="sec-8.3", proposed_text="" if operation == "delete" else "New text.",
                  explanation="x")
    revised = apply_draft(doc, draft)
    for c in revised.clauses:
        assert revised.text[c.start:c.end] == c.text
    if expect_ids:
        assert expect_ids([c.id for c in revised.clauses])
    else:
        assert revised.clause("sec-8.3").text == "New text."
