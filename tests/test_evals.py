"""Offline tests for the eval harness: the labels file is well formed, every
eval NDA parses, and the scoring math is right. No API calls."""

import importlib.util
import json
from pathlib import Path

import pytest

from nda_triage.ingest import parse_nda
from nda_triage.models import Finding, PostureCheck, RuleResult, TriageResult, Usage
from nda_triage.rubric import Posture, load_rubric

EVALS = Path(__file__).resolve().parent.parent / "evals"
spec = importlib.util.spec_from_file_location("run_evals", EVALS / "run_evals.py")
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)

LABELS = json.loads((EVALS / "labels.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", LABELS["cases"], ids=lambda c: c["id"])
def test_label_is_well_formed_and_document_parses(case):
    rubric_rules = {r.id for r in load_rubric().rules}
    assert set(case["expected"]["rules"]) == rubric_rules
    Posture(case["posture"])
    worst = max(case["expected"]["rules"].values(), key=run_evals.RANK.get)
    assert case["expected"]["overall"] == worst  # worst rule wins
    doc = parse_nda(EVALS / case["file"])
    assert len(doc.clauses) >= 5
    assert case["party"].split()[0].upper() in doc.text.upper()


def test_eval_set_covers_every_color_and_has_traps():
    overall = {c["expected"]["overall"] for c in LABELS["cases"]}
    assert overall == {"GREEN", "YELLOW", "RED"}
    assert 10 <= len(LABELS["cases"]) <= 15


@pytest.mark.parametrize("expected, predicted, outcome", [
    ("RED", "RED", "match"),
    ("RED", "GREEN", "false_green"),
    ("YELLOW", "GREEN", "false_green"),
    ("RED", "YELLOW", "under_call"),
    ("GREEN", "YELLOW", "over_escalation"),
    ("YELLOW", "RED", "over_escalation"),
])
def test_compare(expected, predicted, outcome):
    assert run_evals.compare(expected, predicted) == outcome


def fake_result(colors: dict, overall: str, consistent=True) -> TriageResult:
    rules = [RuleResult(rule_id=k, color=v, findings=[Finding(pattern_id="p", clause_ids=[], quote="", reason="r",
                                                            color=v, quote_verified=True)])
             for k, v in colors.items()]
    return TriageResult(source_name="x", sha256="0", posture="mutual", party="p", color=overall, rules=rules,
                        posture_check=PostureCheck(stated="mutual", apparent="mutual", consistent=consistent,
                                                   reason=""), usage=Usage())


def test_score_and_aggregate():
    case = {"id": "c1", "expected": {"overall": "RED", "posture_consistent": False,
                                     "rules": {"non_solicit": "RED", "term": "YELLOW", "residuals": "GREEN"}}}
    result = fake_result({"non_solicit": "GREEN", "term": "RED", "residuals": "YELLOW"}, "RED")
    score = run_evals.score_case(case, result)
    assert score["overall"]["outcome"] == "match"
    assert score["rules"]["non_solicit"]["outcome"] == "false_green"
    assert score["rules"]["term"]["outcome"] == "over_escalation"
    assert score["posture_ok"] is False

    agg = run_evals.aggregate([score])
    assert agg["false_greens"] == 1 and agg["over_escalations"] == 2
    assert agg["rule_accuracy"] == 0
    assert "| non_solicit | 0 | 1 |" in run_evals.summary_markdown(agg, [score], Usage(), "test", False, "proposed")
