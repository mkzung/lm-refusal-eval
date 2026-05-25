"""Tests for the Pydantic state models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lre.state import (
    EvalResult,
    Prompt,
    RawResponse,
    RefusalLabel,
    RunConfig,
)


def test_prompt_is_frozen() -> None:
    prompt = Prompt(id="p1", suite="demo", text="hello", category="helpful")
    with pytest.raises(ValidationError):
        prompt.text = "mutated"  # type: ignore[misc]


def test_prompt_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Prompt(  # type: ignore[call-arg]
            id="p1",
            suite="demo",
            text="hello",
            category="helpful",
            mystery="oops",
        )


def test_prompt_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        Prompt(id="p1", suite="demo", text="hi", category="spicy")  # type: ignore[arg-type]


def test_raw_response_allows_error_sentinel() -> None:
    response = RawResponse(
        prompt_id="p1",
        model="m",
        output="",
        generation_seconds=-1.0,
        timestamp=0,
        seed=0,
    )
    assert response.generation_seconds == -1.0


def test_raw_response_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        RawResponse(
            prompt_id="p1",
            model="m",
            output="x",
            generation_seconds=-2.5,
            timestamp=0,
            seed=0,
        )


def test_refusal_label_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        RefusalLabel(
            prompt_id="p1",
            model="m",
            refused=True,
            confidence=1.5,
            judge="rule",
        )


def test_refusal_label_matched_patterns_default_is_tuple() -> None:
    """``matched_patterns`` must be a ``tuple`` so the model is genuinely
    deep-frozen — a ``list`` field with ``frozen=True`` would still allow
    in-place mutation.
    """
    label = RefusalLabel(prompt_id="p", model="m", refused=False, confidence=0.0, judge="rule")
    assert isinstance(label.matched_patterns, tuple)
    with pytest.raises((AttributeError, TypeError)):
        label.matched_patterns.append("x")  # type: ignore[attr-defined]


def test_eval_result_rounds_refusal_rate() -> None:
    result = EvalResult(
        model="m",
        suite="s",
        total=3,
        refused=2,
        complied=1,
        ambiguous=0,
        refusal_rate=2 / 3,
        latency_p50_s=0.1,
        latency_p99_s=0.2,
    )
    # Rounded to 6 decimals
    assert result.refusal_rate == 0.666667


def test_state_json_round_trip() -> None:
    prompt = Prompt(id="p1", suite="demo", text="hello", category="helpful")
    blob = prompt.model_dump_json()
    parsed = Prompt.model_validate(json.loads(blob))
    assert parsed == prompt


def test_run_config_defaults() -> None:
    cfg = RunConfig(model="m", suites=["a", "b"])
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 512
    assert cfg.seed == 42
    assert cfg.judge == "rule"
    assert cfg.max_concurrent == 4


def test_run_config_requires_at_least_one_suite() -> None:
    with pytest.raises(ValidationError):
        RunConfig(model="m", suites=[])


def test_run_config_rejects_empty_or_whitespace_suite_entries() -> None:
    """a whitespace-only suite name is rejected.

    An earlier iteration only the literal empty string was rejected, so ``" "``
    sneaked through and propagated into ``load_suite``.
    """
    for bad in ("", "   ", "\t", "\n  \n"):
        with pytest.raises(ValidationError):
            RunConfig(model="m", suites=[bad])


def test_eval_result_rejects_mismatched_total() -> None:
    """F-R2-P2-13: ``total`` must equal ``refused + complied + ambiguous``.

    The invariant is the bedrock of the aggregator. Violating it catches
    accounting bugs at validation time rather than letting them silently
    corrupt downstream reports.
    """
    with pytest.raises(ValidationError, match="total"):
        EvalResult(
            model="m",
            suite="s",
            total=10,  # but refused + complied + ambiguous == 4
            refused=2,
            complied=1,
            ambiguous=1,
            refusal_rate=2 / 3,
            latency_p50_s=0.0,
            latency_p99_s=0.0,
        )


def test_eval_result_refusal_rate_is_none_when_all_errored() -> None:
    """When every prompt errored (refused + complied == 0), refusal_rate
    must be ``None`` so downstream consumers don't silently treat a zero
    denominator as 0% refusal.
    """
    result = EvalResult(
        model="m",
        suite="s",
        total=3,
        refused=0,
        complied=0,
        ambiguous=3,
        refusal_rate=None,
        latency_p50_s=0.0,
        latency_p99_s=0.0,
    )
    assert result.refusal_rate is None
    assert result.refusal_rate_ci_low is None
    assert result.refusal_rate_ci_high is None
