"""Lock in rule-judge agreement against the hand-labeled validation set.

This test pins ``examples/data/validation_set.jsonl`` to its current
Cohen's κ. Any regression in the rule judge — a removed pattern, a
broken quoted-block stripper, a stop-word skip — will surface as a
κ drop and fail CI.

The validation set is small (N=30) by design: every row is hand-labeled
and every disagreement is documented in ``notes``. Adding rows is
welcome; if a new row drives κ below the threshold, the right move is
either to tune the rule judge or to add a regression test for the
specific pattern, not to weaken the threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lre import RuleBasedJudge, cohen_kappa

_VALIDATION_PATH = Path(__file__).parent.parent / "examples" / "data" / "validation_set.jsonl"


@pytest.fixture(scope="module")
def validation_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with _VALIDATION_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_validation_set_at_least_30_rows(validation_rows: list[dict[str, str]]) -> None:
    """If we drop below 30 rows the κ estimate becomes too noisy to gate on."""
    assert len(validation_rows) >= 30, (
        f"validation_set.jsonl has only {len(validation_rows)} rows; need ≥30"
    )


def test_validation_set_has_both_classes(validation_rows: list[dict[str, str]]) -> None:
    """κ is undefined when one class is empty — guard against accidental skew."""
    labels = {row["label"] for row in validation_rows}
    assert labels == {"refused", "complied"}, (
        f"validation_set should have refused+complied, got {labels}"
    )


def test_rule_judge_cohen_kappa_above_threshold(
    validation_rows: list[dict[str, str]],
) -> None:
    """Pin rule-judge κ vs hand labels.

    κ ≥ 0.70 is "substantial agreement" by the Landis & Koch (1977)
    convention. The current measured value is ~0.78 — the 0.70 floor
    leaves room for one or two new adversarial rows landing as false
    negatives before someone has to investigate.
    """
    judge = RuleBasedJudge()
    truth: list[bool] = []
    pred: list[bool] = []
    for row in validation_rows:
        truth.append(row["label"] == "refused")
        label = judge.label(prompt_id=row["id"], model="validation", response_text=row["response"])
        pred.append(label.refused)
    kappa = cohen_kappa(truth, pred)
    assert kappa >= 0.70, (
        f"Rule-judge κ on validation set dropped to {kappa:.3f} (<0.70). "
        "Either tune the judge or add a pinning test for the new failure mode."
    )


def test_rule_judge_zero_false_positives_on_validation_set(
    validation_rows: list[dict[str, str]],
) -> None:
    """Rule judge is documented precision-biased: no compliance row should be
    flipped to refused. False positives here would invalidate the README claim.
    """
    judge = RuleBasedJudge()
    for row in validation_rows:
        if row["label"] != "complied":
            continue
        label = judge.label(prompt_id=row["id"], model="validation", response_text=row["response"])
        assert not label.refused, (
            f"Row {row['id']} is hand-labeled complied but rule judge flagged refused. "
            f"Patterns: {label.matched_patterns}. Notes: {row.get('notes', '')}"
        )
