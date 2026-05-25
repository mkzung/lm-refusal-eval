"""Evaluate the rule-based judge against a hand-labeled validation set.

This script is the closest the repository gets to "ground truth": 30
prompt-response pairs constructed to span the easy and adversarial
surface that refusal classifiers actually see — clear refusals, clear
compliance, "I will not lie" false-positive bait, refusal phrases inside
quoted spans, mid-text refusals that follow a partial answer, and
prompt-level verdict-shaped phrases that must not propagate to the
response classification.

Run::

    python examples/validate_judge.py

The script prints a confusion matrix, accuracy, and Cohen's kappa
between the rule judge and the hand labels. A kappa above 0.7 is
typically considered substantial agreement.

The validation set itself lives at
``examples/data/validation_set.jsonl`` and is plain JSONL — one
``{prompt, response, label}`` object per line. Add to it freely; the
script picks up new rows on the next run.
"""

from __future__ import annotations

import json
from pathlib import Path

from lre import RawResponse, RuleBasedJudge, cohen_kappa

_DATA_PATH = Path(__file__).parent / "data" / "validation_set.jsonl"


def _load_validation_rows() -> list[dict[str, str]]:
    """Read the JSONL set and return one dict per row."""
    rows: list[dict[str, str]] = []
    with _DATA_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _judge_predict(judge: RuleBasedJudge, row: dict[str, str]) -> bool:
    """Run the judge over a single row, returning True iff classified as refused."""
    raw = RawResponse(
        prompt_id=row["id"],
        model="hand-labeled-validation",
        output=row["response"],
        generation_seconds=0.0,
        timestamp=0,
        seed=0,
    )
    label = judge.label(prompt_id=raw.prompt_id, model=raw.model, response_text=raw.output)
    return label.refused


def main() -> int:
    rows = _load_validation_rows()
    judge = RuleBasedJudge()

    truth: list[bool] = []
    pred: list[bool] = []
    disagreements: list[tuple[str, bool, bool, str]] = []

    for row in rows:
        truth_refused = row["label"] == "refused"
        pred_refused = _judge_predict(judge, row)
        truth.append(truth_refused)
        pred.append(pred_refused)
        if truth_refused != pred_refused:
            disagreements.append((row["id"], truth_refused, pred_refused, row.get("notes", "")))

    n = len(rows)
    correct = sum(1 for t, p in zip(truth, pred, strict=True) if t == p)
    accuracy = correct / n if n else 0.0
    kappa = cohen_kappa(truth, pred)

    # 2x2 confusion matrix on the binary refused/complied axis.
    cells: dict[tuple[bool, bool], int] = {
        (False, False): 0,
        (False, True): 0,
        (True, False): 0,
        (True, True): 0,
    }
    for t, p in zip(truth, pred, strict=True):
        cells[(t, p)] += 1

    print(f"Validation set: {_DATA_PATH}")
    print(f"N = {n}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{n})")
    print(f"Cohen's kappa: {kappa:.3f}")
    print()
    print("Confusion matrix (rows = truth, cols = prediction):")
    print("                 pred=complied  pred=refused")
    print(f"  truth=complied  {cells[(False, False)]:>13d}  {cells[(False, True)]:>12d}")
    print(f"  truth=refused   {cells[(True, False)]:>13d}  {cells[(True, True)]:>12d}")
    print()

    if disagreements:
        print(f"Disagreements ({len(disagreements)}):")
        for pid, t, p, notes in disagreements:
            t_str = "refused" if t else "complied"
            p_str = "refused" if p else "complied"
            print(f"  {pid}: truth={t_str}, pred={p_str}  -- {notes}")
    else:
        print("No disagreements.")

    # Non-zero exit on substantial disagreement — useful in CI to gate
    # judge changes that silently regress kappa.
    return 0 if kappa >= 0.7 else 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
