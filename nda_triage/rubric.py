"""Load and validate playbook/rubric.yaml.

The YAML is written by a lawyer, not a programmer, so it is validated strictly:
unknown keys are rejected (catches typos like `aplies_to`), every pattern must
have a color for all three postures, and every fallback must point at patterns
that exist. A bad rubric fails at load time with a readable error.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

DEFAULT_RUBRIC = Path(__file__).resolve().parent.parent / "playbook" / "rubric.yaml"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Color(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class Posture(str, Enum):
    discloser = "discloser"
    recipient = "recipient"
    mutual = "mutual"


class Pattern(Strict):
    id: str
    description: str
    colors: dict[Posture, Color]

    @model_validator(mode="after")
    def every_posture_colored(self):
        missing = set(Posture) - set(self.colors)
        if missing:
            raise ValueError(f"pattern {self.id!r} has no color for {sorted(p.value for p in missing)}")
        return self


class Modifier(Strict):
    id: str
    condition: str
    effect: Literal["bump_one_color", "set_red"]
    postures: list[Posture] = list(Posture)


class Fallback(Strict):
    id: str
    applies_to: list[str]
    postures: list[Posture]
    action: Literal["replace", "strike"]
    text: str
    notes: str | None = None


class Rule(Strict):
    id: str
    name: str
    description: str
    patterns: list[Pattern]
    modifiers: list[Modifier] = []
    fallbacks: list[Fallback] = []
    notes: str | None = None

    @model_validator(mode="after")
    def fallbacks_point_at_real_patterns(self):
        pattern_ids = [p.id for p in self.patterns]
        if len(pattern_ids) != len(set(pattern_ids)):
            raise ValueError(f"rule {self.id!r} has duplicate pattern ids")
        for fb in self.fallbacks:
            unknown = set(fb.applies_to) - set(pattern_ids)
            if unknown:
                raise ValueError(f"fallback {fb.id!r} applies to unknown patterns {sorted(unknown)}")
        return self


class Rubric(Strict):
    version: int
    postures: dict[Posture, str]
    colors: dict[Color, str]
    aggregation: Literal["worst_rule_wins"]
    posture_cross_check: bool
    rules: list[Rule]

    @model_validator(mode="after")
    def ids_unique(self):
        # Fallback ids become document ids in the retrieval index (Stage 3),
        # so they must be unique across the whole rubric, not just per rule.
        for kind, ids in [
            ("rule", [r.id for r in self.rules]),
            ("fallback", [f.id for r in self.rules for f in r.fallbacks]),
        ]:
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"duplicate {kind} ids: {sorted(dupes)}")
        return self

    def rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)


def load_rubric(path: str | Path = DEFAULT_RUBRIC) -> Rubric:
    with open(path, encoding="utf-8") as f:
        return Rubric.model_validate(yaml.safe_load(f))
