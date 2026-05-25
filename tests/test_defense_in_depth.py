"""Tests for :mod:`lre.defense_in_depth`.

The paired-defense metric is the v0.5 feature most directly aimed at
the FAR.AI "Defense in Depth" line of work: it quantifies how an
outer classifier changes the effective refusal rate when stacked on
top of the model's own refusal signal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from click.testing import CliRunner

from lre.cli import main
from lre.defense_in_depth import (
    PairedDefense,
    aggregate_paired_results,
    paired_label_responses,
)
from lre.judge import RuleBasedJudge
from lre.state import RawResponse, RefusalLabel


class _FixedJudge:
    """Test judge that always returns a fixed refusal verdict."""

    def __init__(self, refused: bool, kind: str = "custom") -> None:
        self.kind = kind  # type: ignore[assignment]
        self._refused = refused

    def label(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
        return RefusalLabel(
            prompt_id=prompt_id,
            model=model,
            refused=self._refused,
            confidence=0.5,
            judge="manual",
        )

    async def alabel(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
        return self.label(prompt_id, model, response_text)


def _make_responses(n: int) -> list[RawResponse]:
    return [
        RawResponse(
            prompt_id=f"p{i:03d}",
            model="m",
            output=f"text {i}",
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        )
        for i in range(n)
    ]


def test_paired_defense_alabel_returns_inner_outer_joint() -> None:
    inner = _FixedJudge(refused=True)
    outer = _FixedJudge(refused=False)
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    response = _make_responses(1)[0]
    inner_label, outer_label, system_refused = asyncio.run(defense.alabel(response))
    assert inner_label.refused is True
    assert outer_label.refused is False
    assert system_refused is True  # OR of the two


def test_paired_defense_all_inner_refused_no_outer_change() -> None:
    """If the inner judge refuses everything, the outer is irrelevant for the joint."""
    inner = _FixedJudge(refused=True)
    outer = _FixedJudge(refused=False)
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    responses = _make_responses(10)
    inner_labels, outer_labels, joint, ambiguous = asyncio.run(
        paired_label_responses(defense, responses)
    )
    stats = aggregate_paired_results(inner_labels, outer_labels, joint, ambiguous)
    assert stats["inner_refusal_rate"] == 1.0
    assert stats["outer_refusal_rate"] == 0.0
    assert stats["joint_refusal_rate"] == 1.0
    assert stats["delta_vs_inner_only"] == 0.0
    # Legacy aliases still populated.
    assert stats["model_refusal_rate"] == 1.0
    assert stats["classifier_refusal_rate"] == 0.0
    assert stats["delta_vs_model_only"] == 0.0


def test_paired_defense_outer_catches_what_inner_missed() -> None:
    """When inner complies and outer refuses, the joint refusal rate rises."""
    inner = _FixedJudge(refused=False)
    outer = _FixedJudge(refused=True)
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    responses = _make_responses(5)
    inner_labels, outer_labels, joint, ambiguous = asyncio.run(
        paired_label_responses(defense, responses)
    )
    stats = aggregate_paired_results(inner_labels, outer_labels, joint, ambiguous)
    assert stats["inner_refusal_rate"] == 0.0
    assert stats["outer_refusal_rate"] == 1.0
    assert stats["joint_refusal_rate"] == 1.0
    assert stats["delta_vs_inner_only"] == 1.0


def test_paired_defense_joint_refusal_rate_geq_max_individual() -> None:
    """The OR-aggregation invariant: joint >= max(inner, outer)."""

    # Mixed inner / outer signals - we craft an alternation.
    class _AlternatingJudge:
        def __init__(self, mod: int):
            self.kind = "custom"
            self.mod = mod
            self._i = -1

        def _next(self) -> bool:
            self._i += 1
            return (self._i % self.mod) == 0

        def label(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
            return RefusalLabel(
                prompt_id=prompt_id,
                model=model,
                refused=self._next(),
                confidence=0.5,
                judge="manual",
            )

        async def alabel(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
            return self.label(prompt_id, model, response_text)

    inner = _AlternatingJudge(2)
    outer = _AlternatingJudge(3)
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    responses = _make_responses(20)
    inner_labels, outer_labels, joint, ambiguous = asyncio.run(
        paired_label_responses(defense, responses)
    )
    stats = aggregate_paired_results(inner_labels, outer_labels, joint, ambiguous)
    assert isinstance(stats["inner_refusal_rate"], float)
    assert isinstance(stats["outer_refusal_rate"], float)
    assert isinstance(stats["joint_refusal_rate"], float)
    assert stats["joint_refusal_rate"] >= stats["inner_refusal_rate"]
    assert stats["joint_refusal_rate"] >= stats["outer_refusal_rate"]


def test_aggregate_paired_results_empty() -> None:
    stats = aggregate_paired_results([], [], [])
    assert stats["total"] == 0
    assert stats["ambiguous"] == 0
    assert stats["inner_refusal_rate"] == 0.0


def test_aggregate_paired_results_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="length mismatch"):
        aggregate_paired_results(
            [],
            [
                RefusalLabel(
                    prompt_id="p",
                    model="m",
                    refused=False,
                    confidence=0.0,
                    judge="rule",
                )
            ],
            [],
        )


def test_paired_aggregator_excludes_error_sentinels_from_denominator() -> None:
    """F-R4-P1-1: error sentinels must not register as 'inner complied'.

    Constructs a batch of 5 responses where 2 are runner error sentinels
    (``generation_seconds == -1``). With the old aggregator, those rows
    would flow into the inner judge and silently register as
    ``refused=False`` for both inner and outer, dragging the joint
    refusal rate down. The fix: skip sentinels in
    ``paired_label_responses`` and report them under ``ambiguous``.
    """
    real = [
        RawResponse(
            prompt_id=f"ok-{i}",
            model="m",
            output="I cannot help with that.",
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        )
        for i in range(3)
    ]
    errored = [
        RawResponse(
            prompt_id=f"err-{i}",
            model="m",
            output="",
            generation_seconds=-1.0,
            timestamp=0,
            seed=42,
        )
        for i in range(2)
    ]
    responses = real + errored
    inner = _FixedJudge(refused=True)
    outer = _FixedJudge(refused=False)
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    inner_labels, outer_labels, joint, ambiguous = asyncio.run(
        paired_label_responses(defense, responses)
    )
    # Errored rows never reach the judges.
    assert len(inner_labels) == 3
    assert len(outer_labels) == 3
    assert len(joint) == 3
    assert ambiguous == 2
    stats = aggregate_paired_results(inner_labels, outer_labels, joint, ambiguous)
    assert stats["total"] == 5
    assert stats["ambiguous"] == 2
    # Denominator = total - ambiguous = 3, all of which refused inner.
    assert stats["inner_refused"] == 3
    assert stats["inner_refusal_rate"] == 1.0
    assert stats["joint_refusal_rate"] == 1.0
    # If we had naively counted error sentinels as "not refused", the
    # rate would have been 3/5 = 0.6 — a 40-point downward bias.
    assert stats["joint_refusal_rate"] != 0.6


def test_paired_aggregator_all_errored_keeps_rates_zero() -> None:
    """Every prompt errored — judged set is empty; rates stay at 0.0 and ambiguous matches total."""
    errored = [
        RawResponse(
            prompt_id=f"err-{i}",
            model="m",
            output="",
            generation_seconds=-1.0,
            timestamp=0,
            seed=42,
        )
        for i in range(4)
    ]
    inner = _FixedJudge(refused=True)
    outer = _FixedJudge(refused=True)
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    inner_labels, outer_labels, joint, ambiguous = asyncio.run(
        paired_label_responses(defense, errored)
    )
    assert ambiguous == 4
    assert inner_labels == []
    stats = aggregate_paired_results(inner_labels, outer_labels, joint, ambiguous)
    assert stats["total"] == 4
    assert stats["ambiguous"] == 4
    assert stats["joint_refusal_rate"] == 0.0
    assert stats["inner_refusal_rate"] == 0.0
    assert stats["outer_refusal_rate"] == 0.0


def test_did_works_with_rule_judge_as_inner_and_outer() -> None:
    """Sanity check that ``PairedDefense`` integrates cleanly with the real RuleBasedJudge."""
    inner = RuleBasedJudge()
    outer = RuleBasedJudge()
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    # A clearly-refusing response.
    response = RawResponse(
        prompt_id="p1",
        model="m",
        output="I'm sorry, but I can't help with that request.",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    inner_label, outer_label, joint = asyncio.run(defense.alabel(response))
    assert inner_label.refused is True
    assert outer_label.refused is True
    assert joint is True


def test_cli_did_subcommand_end_to_end_json(tmp_path: Path) -> None:
    """``lre did --format json`` writes the full report (summary + per-prompt).

    Uses ``--inner-judge rule --outer-judge rule`` for a deterministic
    local-only run. The inner==outer pair emits a stderr WARNING about
    needing orthogonal judges; the report itself is still well-formed.
    """
    raw_path = tmp_path / "raw.jsonl"
    rr = RawResponse(
        prompt_id="p1",
        model="fake-1b",
        output="I'm sorry, but I can't help with that request.",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    raw_path.write_text(json.dumps(rr.model_dump(mode="json"), sort_keys=True) + "\n")
    out = tmp_path / "did.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "did",
            "--responses",
            str(raw_path),
            "--inner-judge",
            "rule",
            "--outer-judge",
            "rule",
            "--format",
            "json",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Defense-in-depth" in result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in report
    assert report["summary"]["total"] == 1
    # New canonical field names.
    assert "inner_refusal_rate" in report["summary"]
    assert "joint_refusal_rate" in report["summary"]


def test_cli_did_default_format_is_markdown(tmp_path: Path) -> None:
    """``lre did`` with no --format flag prints a Markdown table to stdout.

    F-R4-P2-13: the default output is human-readable Markdown so a
    researcher can eyeball the joint refusal rate without piping
    through ``jq``.
    """
    raw_path = tmp_path / "raw.jsonl"
    rr = RawResponse(
        prompt_id="p1",
        model="fake-1b",
        output="I'm sorry, but I can't help with that request.",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    raw_path.write_text(json.dumps(rr.model_dump(mode="json"), sort_keys=True) + "\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "did",
            "--responses",
            str(raw_path),
            "--inner-judge",
            "rule",
            "--outer-judge",
            "rule",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# Defense-in-depth refusal rates" in result.output
    assert "| Layer |" in result.output
    assert "Joint (inner OR outer)" in result.output


def test_cli_did_warns_when_inner_equals_outer(tmp_path: Path) -> None:
    """F-R4-P2-12: identical inner/outer judges trigger a stderr warning."""
    raw_path = tmp_path / "raw.jsonl"
    rr = RawResponse(
        prompt_id="p1",
        model="fake-1b",
        output="here you go",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    raw_path.write_text(json.dumps(rr.model_dump(mode="json"), sort_keys=True) + "\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "did",
            "--responses",
            str(raw_path),
            "--inner-judge",
            "rule",
            "--outer-judge",
            "rule",
        ],
    )
    assert result.exit_code == 0
    # Click 8.2+ exposes a separate stderr stream on the Result. The
    # warning is emitted via ``click.echo(..., err=True)``.
    assert "WARNING" in result.stderr
    assert "orthogonal" in result.stderr


def test_cli_did_json_requires_out(tmp_path: Path) -> None:
    """``lre did --format json`` errors out cleanly when --out is missing."""
    raw_path = tmp_path / "raw.jsonl"
    rr = RawResponse(
        prompt_id="p1",
        model="fake-1b",
        output="hello",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    raw_path.write_text(json.dumps(rr.model_dump(mode="json"), sort_keys=True) + "\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "did",
            "--responses",
            str(raw_path),
            "--inner-judge",
            "rule",
            "--outer-judge",
            "llm",
            "--format",
            "json",
        ],
    )
    assert result.exit_code != 0
    assert "--format json requires --out" in result.output


def test_to_markdown_did_renders_expected_table() -> None:
    """Pure-function check on :func:`to_markdown_did`."""
    from lre.defense_in_depth import to_markdown_did

    summary = {
        "total": 10,
        "ambiguous": 2,
        "inner_refused": 3,
        "outer_refused": 4,
        "joint_refused": 5,
        "inner_refusal_rate": 0.375,
        "outer_refusal_rate": 0.5,
        "joint_refusal_rate": 0.625,
        "delta_vs_inner_only": 0.25,
    }
    md = to_markdown_did(summary)
    assert "# Defense-in-depth refusal rates" in md
    assert "Total responses: 10 (judged: 8, ambiguous/errored: 2)" in md
    assert "| Inner (model-side) | 3 | 0.3750 |" in md
    assert "| Outer (classifier-side) | 4 | 0.5000 |" in md
    assert "| Joint (inner OR outer) | 5 | 0.6250 |" in md
    assert "Δ joint vs inner-only: +0.2500" in md


def test_cli_did_help_mentions_far_ai_research() -> None:
    """The ``--help`` text must point at FAR.AI's layered-defenses research.

    We no longer claim a specific paper title we cannot verify; the
    citation now points at https://www.far.ai/news (the canonical
    landing for layered-defense work including the STACK pipeline
    attacks paper) and acknowledges that the literature finds
    vulnerabilities in stacked refusal classifiers.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["did", "--help"])
    assert result.exit_code == 0
    assert "FAR.AI" in result.output


def test_paired_label_responses_fans_out_concurrently() -> None:
    """v0.7/v0.8 (P1-20): pinned timing on paired-defense fan-out.

    Eight responses, each judge call sleeps 100ms — with
    ``max_concurrent=4`` the total wall time must be well under 400ms.
    The pre-v0.7 serial implementation took ~1.6s (8 * 2 judges * 100ms).
    Generous tolerance to absorb CI noise.
    """
    import time as _time

    class _SlowJudge:
        kind = "custom"

        def label(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
            return RefusalLabel(
                prompt_id=prompt_id,
                model=model,
                refused=False,
                heuristic_score=0.0,
                judge="manual",
            )

        async def alabel(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
            await asyncio.sleep(0.1)
            return self.label(prompt_id, model, response_text)

    inner = _SlowJudge()
    outer = _SlowJudge()
    defense = PairedDefense(inner_judge=inner, outer_judge=outer)
    responses = _make_responses(8)
    start = _time.perf_counter()
    asyncio.run(paired_label_responses(defense, responses, max_concurrent=4))
    elapsed = _time.perf_counter() - start
    # 8 items at 100ms each, max_concurrent=4 ⇒ ~2 batches => ~200ms.
    # The two judges run SEQUENTIALLY inside ``defense.alabel`` so each
    # item is ~200ms; with concurrency 4, total ≈ 400ms. Allow generous
    # CI slack while still failing on serial-mode regressions (~1.6s).
    assert elapsed < 0.8, (
        f"paired_label_responses serial regression: {elapsed:.3f}s "
        f"(expected ~0.4s under max_concurrent=4)"
    )
