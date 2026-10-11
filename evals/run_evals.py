"""Stage 6: the eval harness.

Runs every labeled NDA in evals/labels.json through the pipeline and compares
the result with the attorney's expected verdicts.

    python evals/run_evals.py                 # classification only (cheaper)
    python evals/run_evals.py --redlines      # also draft and check redlines
    python evals/run_evals.py --cases 04 06   # only cases whose id starts with these

What it reports:

- Overall verdict accuracy, and per-rule accuracy.
- False GREENs, the dangerous error: the attorney expected YELLOW or RED and
  the tool said GREEN, so an NDA that needed a lawyer could be signed
  without one. Reported per rule and for the overall verdict.
- Under-calls (RED called YELLOW) and over-escalations (GREEN called YELLOW
  or RED). Over-escalation is the cost side: it wastes counsel's time.
- Posture cross-check accuracy, unverified quotes, needs_review counts.
- With --redlines: redline pass rate and how often unfiltered vector search
  would have picked the fallback that exact lookup chose.

Every disagreement is written to disagreements.jsonl with the model's
findings and reasons, so the attorney can decide whether the model or the
rubric (or the label) is wrong.

Results go to evals/results/<timestamp>/ (gitignored); the summary is also
copied to evals/RESULTS.md, which is committed.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from nda_triage.agent import review  # noqa: E402
from nda_triage.classify import estimated_cost, total, triage  # noqa: E402
from nda_triage.ingest import parse_nda  # noqa: E402
from nda_triage.models import TriageResult  # noqa: E402
from nda_triage.rubric import Posture, load_rubric  # noqa: E402

RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


# ---------------------------------------------------------------------------
# Scoring: pure functions, tested offline
# ---------------------------------------------------------------------------

def compare(expected: str, predicted: str) -> str:
    """'match', 'false_green', 'under_call', or 'over_escalation'."""
    if expected == predicted:
        return "match"
    if predicted == "GREEN":
        return "false_green"
    return "under_call" if RANK[predicted] < RANK[expected] else "over_escalation"


def score_case(case: dict, result: TriageResult) -> dict:
    exp = case["expected"]
    predicted_rules = {r.rule_id: r.color for r in result.rules}
    rules = {rule_id: {"expected": color, "predicted": predicted_rules.get(rule_id, "MISSING"),
                       "outcome": compare(color, predicted_rules.get(rule_id, "GREEN"))}
             for rule_id, color in exp["rules"].items()}
    posture_ok = None
    if result.posture_check is not None and "posture_consistent" in exp:
        posture_ok = result.posture_check.consistent == exp["posture_consistent"]
    return {
        "id": case["id"],
        "overall": {"expected": exp["overall"], "predicted": result.color,
                    "outcome": compare(exp["overall"], result.color)},
        "rules": rules,
        "posture_ok": posture_ok,
        "unverified_quotes": sum(not f.quote_verified for r in result.rules for f in r.findings),
        "needs_review": [r.rule_id for r in result.rules if r.needs_review],
        "redlines": [{"rule_id": rl.rule_id, "status": rl.status, "attempts": rl.check.attempts if rl.check else 0,
                      "retrieval": rl.retrieval, "fallback_id": rl.fallback_id,
                      "semantic_top1": rl.semantic_top[0] if rl.semantic_top else None}
                     for rl in result.redlines],
    }


def aggregate(scores: list[dict]) -> dict:
    overall = Counter(s["overall"]["outcome"] for s in scores)
    per_rule: dict[str, Counter] = defaultdict(Counter)
    for s in scores:
        for rule_id, r in s["rules"].items():
            per_rule[rule_id][r["outcome"]] += 1
    posture = [s["posture_ok"] for s in scores if s["posture_ok"] is not None]
    redlines = [rl for s in scores for rl in s["redlines"]]
    exact = [rl for rl in redlines if rl["retrieval"] == "exact"]
    return {
        "cases": len(scores),
        "overall": dict(overall),
        "overall_accuracy": overall["match"] / len(scores) if scores else 0.0,
        "per_rule": {k: dict(v) for k, v in per_rule.items()},
        "rule_cells": sum(sum(v.values()) for v in per_rule.values()),
        "rule_accuracy": (sum(v["match"] for v in per_rule.values()) /
                          max(1, sum(sum(v.values()) for v in per_rule.values()))),
        "false_greens": sum(v["false_green"] for v in per_rule.values()),
        "under_calls": sum(v["under_call"] for v in per_rule.values()),
        "over_escalations": sum(v["over_escalation"] for v in per_rule.values()),
        "posture_check_accuracy": sum(posture) / len(posture) if posture else None,
        "unverified_quotes": sum(s["unverified_quotes"] for s in scores),
        "needs_review": sum(len(s["needs_review"]) for s in scores),
        "redlines": len(redlines),
        "redlines_passed": sum(rl["status"] == "passed" for rl in redlines),
        "retrieval_agreement": (sum(rl["semantic_top1"] == rl["fallback_id"] for rl in exact) / len(exact)
                                if exact else None),
    }


def disagreements(case: dict, score: dict, result: TriageResult) -> list[dict]:
    rows = []
    by_rule = {r.rule_id: r for r in result.rules}
    for rule_id, r in score["rules"].items():
        if r["outcome"] == "match":
            continue
        rr = by_rule.get(rule_id)
        rows.append({
            "case": case["id"], "rule": rule_id, "posture": case["posture"],
            "expected": r["expected"], "predicted": r["predicted"], "outcome": r["outcome"],
            "findings": [f.model_dump() for f in rr.findings] if rr else [],
            "modifiers_applied": rr.modifiers_applied if rr else [],
            "needs_review": rr.needs_review if rr else None,
        })
    return rows


def summary_markdown(agg: dict, scores: list[dict], usage, run_id: str, with_redlines: bool, label_status: str,
                     unconfirmed: tuple = ()) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    lines = [
        f"# Eval results ({run_id})",
        "",
        f"Labels: **{label_status}**. Mode: {'classification + redlines' if with_redlines else 'classification only'}. "
        f"{agg['cases']} synthetic NDAs, {agg['rule_cells']} rule verdicts.",
        "",
        *([f"Labels not yet confirmed by the attorney: {', '.join(unconfirmed)}.", ""] if unconfirmed else []),
        "| Metric | Result |",
        "|---|---|",
        f"| Overall verdict accuracy | {pct(agg['overall_accuracy'])} ({agg['overall'].get('match', 0)}/{agg['cases']}) |",
        f"| Overall false GREENs | {agg['overall'].get('false_green', 0)} |",
        f"| Rule verdict accuracy | {pct(agg['rule_accuracy'])} |",
        f"| Rule false GREENs (dangerous) | {agg['false_greens']} |",
        f"| Rule under-calls (RED as YELLOW) | {agg['under_calls']} |",
        f"| Rule over-escalations | {agg['over_escalations']} |",
        f"| Posture cross-check accuracy | {pct(agg['posture_check_accuracy'])} |",
        f"| Unverified quotes | {agg['unverified_quotes']} |",
        f"| Rules marked needs_review | {agg['needs_review']} |",
    ]
    if with_redlines:
        lines += [f"| Redlines passed self-check | {agg['redlines_passed']}/{agg['redlines']} |",
                  f"| Vector search top-1 = exact fallback | {pct(agg['retrieval_agreement'])} |"]
    lines += [f"| API calls / estimated cost | {usage.calls} / ${estimated_cost(usage):.2f} |", "",
              "## Per rule", "", "| Rule | Match | False GREEN | Under-call | Over-escalation |", "|---|---|---|---|---|"]
    for rule_id, c in agg["per_rule"].items():
        lines.append(f"| {rule_id} | {c.get('match', 0)} | {c.get('false_green', 0)} | "
                     f"{c.get('under_call', 0)} | {c.get('over_escalation', 0)} |")
    lines += ["", "## Per case", "", "| Case | Expected | Predicted | Rule misses |", "|---|---|---|---|"]
    for s in scores:
        misses = ", ".join(f"{k} ({v['expected']} vs {v['predicted']})"
                           for k, v in s["rules"].items() if v["outcome"] != "match") or "none"
        lines.append(f"| {s['id']} | {s['overall']['expected']} | {s['overall']['predicted']} | {misses} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NDA triage evals.")
    parser.add_argument("--labels", default=str(HERE / "labels.json"))
    parser.add_argument("--cases", nargs="*", help="only run case ids starting with these prefixes")
    parser.add_argument("--redlines", action="store_true", help="also draft and self-check redlines (more calls)")
    parser.add_argument("--allow-proposed", action="store_true", help="score against labels the attorney hasn't confirmed")
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if labels["status"] != "confirmed" and not args.allow_proposed:
        sys.exit("labels.json is not confirmed by the attorney yet. Confirm it, or pass --allow-proposed.")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    cases = [c for c in labels["cases"] if not args.cases or any(c["id"].startswith(p) for p in args.cases)]
    rubric = load_rubric()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True)

    scores, usages, raw = [], [], []
    with open(out_dir / "disagreements.jsonl", "w", encoding="utf-8") as dis:
        for case in cases:
            started = time.time()
            doc = parse_nda(HERE / case["file"])
            run = review if args.redlines else triage
            result = run(doc, rubric, Posture(case["posture"]), case["party"])
            score = score_case(case, result)
            scores.append(score)
            usages.append(result.usage)
            raw.append(json.loads(result.model_dump_json()))
            for row in disagreements(case, score, result):
                dis.write(json.dumps(row) + "\n")
            mark = "ok " if score["overall"]["outcome"] == "match" else "XX "
            print(f"{mark}{case['id']:<32} expected {score['overall']['expected']:<6} got {result.color:<6} "
                  f"({time.time() - started:.0f}s)")

    usage = total(usages)
    agg = aggregate(scores)
    summary = summary_markdown(agg, scores, usage, run_id, args.redlines, labels["status"],
                               tuple(labels.get("unconfirmed_cases", [])))
    (out_dir / "results.json").write_text(json.dumps({"aggregate": agg, "scores": scores, "results": raw}, indent=2),
                                          encoding="utf-8")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    if not args.cases:  # only a full run updates the committed summary
        (HERE / "RESULTS.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Details: {out_dir}")


if __name__ == "__main__":
    main()
