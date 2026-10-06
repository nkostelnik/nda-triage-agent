import copy

import pytest
import yaml
from pydantic import ValidationError

from nda_triage.rubric import DEFAULT_RUBRIC, Color, Posture, Rubric, load_rubric


@pytest.fixture
def raw():
    with open(DEFAULT_RUBRIC, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_playbook_rubric_loads():
    rubric = load_rubric()
    assert len(rubric.rules) == 9
    assert rubric.rule("residuals").patterns[1].colors[Posture.discloser] == Color.RED


def test_fallback_pointing_at_missing_pattern_is_rejected(raw):
    bad = copy.deepcopy(raw)
    bad["rules"][0]["fallbacks"][0]["applies_to"].append("no_such_pattern")
    with pytest.raises(ValidationError, match="unknown patterns"):
        Rubric.model_validate(bad)


def test_pattern_missing_a_posture_is_rejected(raw):
    bad = copy.deepcopy(raw)
    del bad["rules"][0]["patterns"][0]["colors"]["mutual"]
    with pytest.raises(ValidationError, match="no color for"):
        Rubric.model_validate(bad)


def test_typo_in_key_is_rejected(raw):
    bad = copy.deepcopy(raw)
    fb = bad["rules"][0]["fallbacks"][0]
    fb["aplies_to"] = fb.pop("applies_to")
    with pytest.raises(ValidationError):
        Rubric.model_validate(bad)
