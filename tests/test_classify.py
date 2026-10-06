"""Classification tests run against a fake API client, so they are free,
fast, and deterministic. They test everything except the model's judgment:
the prompt shape, caching setup, schema, scoring, verification, and failure
handling. The model's judgment is what the Stage 6 evals measure."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from nda_triage.classify import assessment_schema, render_document, triage
from nda_triage.ingest import parse_nda
from nda_triage.rubric import Posture, load_rubric

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_nda.txt"
PARTY = "Acme Analytics, Inc."

# What a correct model would say about the sample NDA, rule by rule.
SAMPLE_ANSWERS = {
    "mutuality": {"findings": [{"pattern_id": "fully_mutual", "clause_ids": ["sec-2.1"],
                                "quote": "disclosed by either party to the other", "reason": "Both sides protected."}]},
    "term": {"findings": [{"pattern_id": "term_2_to_3yr", "clause_ids": ["sec-4"],
                           "quote": "continues for two (2) years", "reason": "Two-year term."}],
             "triggered_modifiers": []},
    "survival": {"findings": [{"pattern_id": "survival_2_to_3yr", "clause_ids": ["sec-5"],
                               "quote": "survive for three (3) years", "reason": "Three-year survival."}],
                 "triggered_modifiers": [{"modifier_id": "no_trade_secret_carveout", "reason": "No carve-out."}]},
    "non_solicit": {"findings": [{"pattern_id": "narrow_employee_non_solicit", "clause_ids": ["sec-8.3"],
                                  "quote": "neither party will solicit for employment any employee",
                                  "reason": "Employee non-solicit hidden under Cooperation."}]},
    "non_compete": {"findings": [{"pattern_id": "no_non_compete", "clause_ids": [], "quote": "", "reason": "None."}]},
    "ip_assignment": {"findings": [{"pattern_id": "express_no_license", "clause_ids": ["sec-6"],
                                    "quote": "Nothing in this Agreement grants either party any license",
                                    "reason": "Express no-license clause."}]},
    "residuals": {"findings": [{"pattern_id": "no_residuals", "clause_ids": [], "quote": "", "reason": "None."}]},
    "indemnification": {"findings": [{"pattern_id": "no_indemnity", "clause_ids": [], "quote": "", "reason": "None."}]},
    "ci_definition": {"findings": [{"pattern_id": "broad_definition", "clause_ids": ["sec-2.1"],
                                    "quote": "whether or not marked as confidential", "reason": "Broad."}]},
}


class FakeClient:
    """Stands in for anthropic.Anthropic(). Answers each call from a table
    keyed by rule id, and records every request it receives."""

    def __init__(self, answers, posture="mutual", refuse=()):
        self.answers, self.posture, self.refuse = answers, posture, set(refuse)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        question = kwargs["messages"][0]["content"][1]["text"]
        m = re.search(r'<rule id="(\w+)">', question)
        usage = SimpleNamespace(input_tokens=50, cache_creation_input_tokens=0 if len(self.requests) > 1 else 900,
                                cache_read_input_tokens=900 if len(self.requests) > 1 else 0, output_tokens=40)
        if m and m.group(1) in self.refuse:
            return SimpleNamespace(parsed_output=None, stop_reason="refusal", usage=usage, model="claude-sonnet-5-5")
        data = self.answers[m.group(1)] if m else {"apparent_posture": self.posture, "reason": "Both protected."}
        return SimpleNamespace(parsed_output=kwargs["output_format"].model_validate(data),
                               stop_reason="end_turn", usage=usage, model="claude-sonnet-5-5")


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


@pytest.fixture(scope="module")
def doc():
    return parse_nda(SAMPLE)


def run(doc, rubric, answers=SAMPLE_ANSWERS, posture=Posture.mutual, **fake_kwargs):
    client = FakeClient(answers, **fake_kwargs)
    return triage(doc, rubric, posture, PARTY, client=client), client


def rule(result, rule_id):
    return next(r for r in result.rules if r.rule_id == rule_id)


def test_buried_non_solicit_makes_the_nda_red(doc, rubric):
    result, _ = run(doc, rubric)
    assert result.color == "RED"
    assert rule(result, "non_solicit").color == "RED"
    assert [r.rule_id for r in result.rules if r.color != "GREEN"] == ["non_solicit"]


def test_colors_come_from_the_rubric_for_the_stated_posture(doc, rubric):
    # Same facts, discloser posture: a 2-3 year survival period is YELLOW,
    # and with no trade-secret carve-out the modifier makes it RED.
    result, _ = run(doc, rubric, posture=Posture.discloser)
    survival = rule(result, "survival")
    assert survival.findings[0].color == "YELLOW"
    assert survival.color == "RED"
    assert survival.modifiers_applied[0].startswith("no_trade_secret_carveout")


def test_modifier_ignored_outside_its_postures(doc, rubric):
    result, _ = run(doc, rubric, posture=Posture.mutual)
    assert rule(result, "survival").modifiers_applied == []


def test_unverifiable_quote_cannot_be_green(doc, rubric):
    answers = dict(SAMPLE_ANSWERS, ip_assignment={"findings": [{
        "pattern_id": "express_no_license", "clause_ids": ["sec-6"],
        "quote": "words that are not in the clause", "reason": "Made up."}]})
    result, _ = run(doc, rubric, answers=answers)
    ip = rule(result, "ip_assignment")
    assert ip.findings[0].quote_verified is False
    assert ip.color == "YELLOW" and ip.needs_review


def test_unknown_clause_id_is_not_verified(doc, rubric):
    answers = dict(SAMPLE_ANSWERS, term={"findings": [{
        "pattern_id": "term_2_to_3yr", "clause_ids": ["sec-99"], "quote": "continues for two (2) years",
        "reason": "Wrong id."}], "triggered_modifiers": []})
    result, _ = run(doc, rubric, answers=answers)
    assert rule(result, "term").findings[0].quote_verified is False


def test_no_findings_fails_closed(doc, rubric):
    answers = dict(SAMPLE_ANSWERS, term={"findings": [], "triggered_modifiers": []})
    result, _ = run(doc, rubric, answers=answers)
    assert rule(result, "term").color == "YELLOW"
    assert "rubric has no position" in rule(result, "term").needs_review


def test_refusal_fails_closed(doc, rubric):
    result, _ = run(doc, rubric, refuse={"indemnification"})
    ind = rule(result, "indemnification")
    assert ind.color == "YELLOW" and "declined" in ind.needs_review


def test_posture_mismatch_is_reported(doc, rubric):
    result, _ = run(doc, rubric, posture=Posture.discloser)
    assert result.posture_check.consistent is False
    assert result.posture_check.apparent == "mutual"


def test_requests_share_a_cacheable_prefix(doc, rubric):
    _, client = run(doc, rubric)
    assert len(client.requests) == 10  # posture check + 9 rules
    first = client.requests[0]
    doc_block = first["messages"][0]["content"][0]
    assert doc_block["cache_control"] == {"type": "ephemeral"}
    for req in client.requests:
        assert req["system"] == first["system"]
        assert req["messages"][0]["content"][0] == doc_block
        assert req["model"] == "claude-sonnet-5-5"


def test_usage_is_totaled(doc, rubric):
    result, _ = run(doc, rubric)
    assert result.usage.calls == 10
    assert result.usage.cache_write_tokens == 900
    assert result.usage.cache_read_tokens == 900 * 9


def test_schema_only_allows_the_rules_own_patterns(rubric):
    schema = assessment_schema(rubric.rule("residuals"))
    with pytest.raises(Exception):
        schema.model_validate({"findings": [{"pattern_id": "fully_mutual", "clause_ids": [], "quote": "", "reason": "x"}]})


def test_document_render_tags_every_clause(doc):
    rendered = render_document(doc)
    assert rendered.count("<clause id=") == len(doc.clauses)
    assert '<clause id="sec-8.3">' in rendered


def test_result_carries_the_review_notice(doc, rubric):
    result, _ = run(doc, rubric)
    assert "attorney review" in result.notice and "Not legal advice" in result.notice
