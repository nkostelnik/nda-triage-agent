"""Retrieval tests use the real local embedding model (downloaded once to
~/.cache/chroma) and a throwaway index directory per test."""

import shutil
from pathlib import Path

import pytest

from nda_triage.retrieval import Retriever, lookup_fallbacks
from nda_triage.rubric import Posture, load_rubric

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


@pytest.fixture
def retriever(rubric, tmp_path):
    return Retriever(rubric, index_dir=tmp_path / "index")


def test_every_fallback_is_indexed(retriever, rubric):
    assert retriever.playbook.count() == sum(len(r.fallbacks) for r in rubric.rules)


def test_index_is_reused_when_rubric_unchanged(rubric, tmp_path):
    first = Retriever(rubric, index_dir=tmp_path / "index").playbook.id
    second = Retriever(rubric, index_dir=tmp_path / "index").playbook.id
    assert first == second


def test_index_is_rebuilt_when_a_fallback_changes(rubric, tmp_path):
    first = Retriever(rubric, index_dir=tmp_path / "index").playbook.id
    edited = rubric.model_copy(deep=True)
    edited.rules[0].fallbacks[0].text += " Edited."
    second = Retriever(edited, index_dir=tmp_path / "index").playbook.id
    assert first != second


@pytest.mark.parametrize("posture", list(Posture))
def test_results_respect_posture(retriever, rubric, posture):
    allowed = {fb.id for r in rubric.rules for fb in r.fallbacks if posture in fb.postures}
    hits = retriever.search("The Receiving Party shall keep information confidential.", posture, k=20)
    assert hits and {h.id for h in hits} <= allowed


def test_rule_filter(retriever):
    hits = retriever.search("any clause at all", Posture.mutual, rule_id="non_compete", k=5)
    assert [h.id for h in hits] == ["non_compete_strike"]


@pytest.mark.parametrize("text, posture, expected", [
    ("The Receiving Party's obligations survive for three (3) years after termination.",
     Posture.discloser, "survival_make_perpetual"),
    ("The Receiving Party's obligations shall survive in perpetuity.",
     Posture.recipient, "survival_3yr"),
    ("This Agreement continues until terminated by either party.",
     Posture.mutual, "term_3yr_30day_notice"),
    ("Recipient may use any information retained in the unaided memory of its employees.",
     Posture.discloser, "residuals_strike"),
])
def test_clear_cases_rank_first(retriever, text, posture, expected):
    assert retriever.search(text, posture)[0].id == expected


def test_exact_lookup(rubric):
    found = lookup_fallbacks(rubric, "survival", "survival_perpetual", Posture.recipient)
    assert [fb.id for fb in found] == ["survival_3yr"]
    assert lookup_fallbacks(rubric, "survival", "survival_perpetual", Posture.mutual) == []


def test_corpus_is_chunked_by_clause(retriever, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(SAMPLES / "sample_nda.txt", corpus)
    assert retriever.index_corpus(corpus) == 14  # 16 clauses minus preamble and signature

    hits = retriever.search_corpus("neither party will solicit for employment any employee of the other party")
    assert hits[0].id == "sample_nda:sec-8.3"


def test_empty_corpus_returns_nothing(retriever, tmp_path):
    assert retriever.index_corpus(tmp_path) == 0
    assert retriever.search_corpus("anything") == []
