"""Tests for the async eval runner."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from lre.judge import LLMJudge
from lre.prompts import load_suite
from lre.runner import (
    _percentile,  # type: ignore[attr-defined]
    aggregate_results,
    ajudge_responses,
    judge_responses,
    run_eval,
)
from lre.state import Prompt, RawResponse, RefusalLabel, RunConfig
from lre.synthetic import FakeModelClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_run_eval_end_to_end_on_harmful_helpful() -> None:
    client = FakeModelClient(name="fake-1b", refusal_rate=0.6, seed=42)
    config = RunConfig(model=client.name, suites=["harmful_helpful"], seed=42)
    responses = _run(run_eval(client, "harmful_helpful", config))
    assert len(responses) == 15
    # Each response has the prompt_id stamped by the runner.
    assert all(r.prompt_id for r in responses)
    assert all(r.model == "fake-1b" for r in responses)


def test_run_eval_is_deterministic() -> None:
    client_a = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=7)
    client_b = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=7)
    config = RunConfig(model="fake-1b", suites=["jailbreak_styles"], seed=11)
    out_a = _run(run_eval(client_a, "jailbreak_styles", config))
    out_b = _run(run_eval(client_b, "jailbreak_styles", config))
    assert [r.output for r in out_a] == [r.output for r in out_b]


def test_run_eval_handles_per_prompt_errors() -> None:
    class ExplodingClient:
        name = "boom"

        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            max_tokens: int,
            seed: int,
        ) -> RawResponse:
            raise RuntimeError("simulated failure")

    config = RunConfig(model="boom", suites=["harmful_helpful"], seed=1)
    responses = _run(run_eval(ExplodingClient(), "harmful_helpful", config))
    assert len(responses) == 15
    # Every entry should be the error sentinel (generation_seconds=-1.0).
    assert all(r.generation_seconds == -1.0 for r in responses)
    assert all(r.output == "" for r in responses)


def test_aggregate_marks_errored_as_ambiguous() -> None:
    client = FakeModelClient(name="fake-1b", refusal_rate=1.0, seed=1)
    config = RunConfig(model="fake-1b", suites=["harmful_helpful"], seed=1)
    responses = _run(run_eval(client, "harmful_helpful", config))
    # Force one error sentinel manually
    poisoned = list(responses)
    poisoned[0] = RawResponse(
        prompt_id=responses[0].prompt_id,
        model=responses[0].model,
        output="",
        generation_seconds=-1.0,
        timestamp=0,
        seed=1,
    )
    labels = judge_responses(poisoned)
    result = aggregate_results(
        model="fake-1b",
        suite="harmful_helpful",
        responses=poisoned,
        labels=labels,
    )
    assert result.ambiguous == 1
    assert result.refused + result.complied == len(poisoned) - 1
    assert 0.0 <= result.refusal_rate <= 1.0


def test_aggregate_rejects_length_mismatch() -> None:
    client = FakeModelClient(name="fake-1b", refusal_rate=0.0, seed=0)
    config = RunConfig(model="fake-1b", suites=["harmful_helpful"], seed=0)
    responses = _run(run_eval(client, "harmful_helpful", config))
    labels = judge_responses(responses)[:-1]
    with pytest.raises(ValueError):
        aggregate_results(
            model="fake-1b",
            suite="harmful_helpful",
            responses=responses,
            labels=labels,
        )


def test_percentile_helper() -> None:
    # Empty list returns 0.0 (used as a "no measurements" sentinel).
    assert _percentile([], 0.5) == 0.0
    # Single-element list always returns that element.
    assert _percentile([1.0], 0.99) == 1.0
    # Nearest-rank, 1-indexed: ceil(q * n) on the sorted list.
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.99) == 5.0
    # ceil(0.5 * 4) = 2; banker's rounding would have given 2 too, but with
    # ceil it's stable.
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    # ceil(0.5 * 6) = 3.
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5) == 3.0
    # Regression cases from the audit:
    # ceil(0.025 * 20) = 1 → first element (banker's round would give 0).
    assert _percentile([float(i) for i in range(1, 21)], 0.025) == 1.0
    # ceil(0.99 * 150) = 149 → 149th element (banker's round would give 150).
    assert _percentile([float(i) for i in range(1, 151)], 0.99) == 149.0


def test_percentile_matches_numpy_inverted_cdf() -> None:
    """v0.9 (P0-4): ``_percentile`` matches ``numpy.percentile(method='inverted_cdf')``.

    The runner docstring previously claimed ``method='lower'`` — wrong.
    ``method='lower'`` uses ``floor(q * (n - 1))`` (zero-indexed),
    whereas ``_percentile`` uses ``ceil(q * n)`` (one-indexed) which
    matches the inverted-CDF rule. Pin the claim against numpy.
    """
    numpy = pytest.importorskip("numpy")

    latencies = [float(i) for i in range(100)]
    arr = numpy.asarray(latencies)
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        ours = _percentile(latencies, q)
        theirs = float(numpy.percentile(arr, q * 100.0, method="inverted_cdf"))
        assert ours == theirs, f"q={q}: ours={ours} vs numpy inverted_cdf={theirs}"

    # And confirm the contrast with method='lower' — they differ.
    # At q=0.50: inverted_cdf rank = ceil(0.5 * 100) = 50 ⇒ values[49] = 49.
    # method='lower' on 100 elements: floor(0.5 * 99) = 49 ⇒ values[49] = 49 (matches here).
    # At q=0.99: inverted_cdf rank = 99 ⇒ values[98] = 98.
    # method='lower' on 100 elements: floor(0.99 * 99) = 98 ⇒ values[98] = 98 (also matches).
    # Use a non-100 length where the conventions visibly diverge.
    short = [float(i) for i in range(10)]
    short_arr = numpy.asarray(short)
    # q=0.5: inverted_cdf rank = ceil(5.0) = 5 ⇒ short[4] = 4.
    # method='lower':        floor(0.5 * 9) = 4 ⇒ short[4] = 4. (Match.)
    # q=0.75: inverted_cdf rank = ceil(7.5) = 8 ⇒ short[7] = 7.
    # method='lower':         floor(0.75 * 9) = 6 ⇒ short[6] = 6. (Differ.)
    assert _percentile(short, 0.75) == 7.0
    assert float(numpy.percentile(short_arr, 75.0, method="inverted_cdf")) == 7.0
    assert float(numpy.percentile(short_arr, 75.0, method="lower")) == 6.0


def test_p50_and_p99_use_same_percentile_method() -> None:
    """v0.8 (P0-5): p50 and p99 must share the nearest-rank definition.

    Pre-v0.8 p50 used ``statistics.median`` (linear interpolation), so on
    even-sized samples it returned a value NOT in the input set, while
    p99 returned an actual sample. The contract is: both percentiles
    yield an element from the input set, computed via ``_percentile``.

    On ``list(range(100))``:
    * ``_percentile(0.50)`` = element at rank ceil(0.50 * 100) = 50 ⇒ value 49.
    * ``_percentile(0.99)`` = element at rank ceil(0.99 * 100) = 99 ⇒ value 98.
    """
    values = [float(i) for i in range(100)]
    p50 = _percentile(values, 0.50)
    p99 = _percentile(values, 0.99)
    assert p50 == 49.0
    assert p99 == 98.0
    # Same family — both are EXACT input values, not interpolations.
    assert p50 in values
    assert p99 in values


def test_aggregate_results_p50_is_nearest_rank() -> None:
    """``aggregate_results`` reports p50 via ``_percentile``, not ``statistics.median``.

    On an even-sized latency vector (4 values), the two definitions
    disagree: ``statistics.median([0.1, 0.2, 0.3, 0.4])`` is 0.25 (the
    interpolated midpoint), but nearest-rank ``_percentile(..., 0.5)``
    is 0.2 (the actual second element). Pin the latter so latency
    reporting is consistent with p99.
    """
    from lre.runner import aggregate_results
    from lre.state import RawResponse, RefusalLabel

    responses = [
        RawResponse(
            prompt_id=f"p{i}",
            model="m",
            output="ok",
            generation_seconds=lat,
            timestamp=0,
            seed=0,
        )
        for i, lat in enumerate([0.1, 0.2, 0.3, 0.4])
    ]
    labels = [
        RefusalLabel(
            prompt_id=f"p{i}",
            model="m",
            refused=False,
            heuristic_score=0.0,
            judge="rule",
        )
        for i in range(4)
    ]
    result = aggregate_results(model="m", suite="s", responses=responses, labels=labels)
    # Nearest-rank p50 = second element = 0.2. statistics.median would
    # have returned 0.25.
    assert result.latency_p50_s == 0.2


def test_judge_responses_llm_requires_instance() -> None:
    """``judge_responses(kind='llm')`` must raise ``ValueError`` when no
    ``llm_judge`` is supplied. Silently falling back to the rule judge —
    as the original implementation did — hides a config bug.
    """
    client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=1)
    config = RunConfig(model="fake-1b", suites=["harmful_helpful"], seed=1)
    responses = _run(run_eval(client, "harmful_helpful", config))
    with pytest.raises(ValueError, match="LLM-judge requires"):
        judge_responses(responses, kind="llm")


def test_aggregate_results_returns_none_when_all_errored() -> None:
    """If every generation errored, refusal_rate should be None — not 0.0."""
    responses = [
        RawResponse(
            prompt_id=f"p{i}",
            model="m",
            output="",
            generation_seconds=-1.0,
            timestamp=0,
            seed=0,
        )
        for i in range(3)
    ]
    labels = judge_responses(responses)
    result = aggregate_results(model="m", suite="s", responses=responses, labels=labels)
    assert result.refusal_rate is None
    assert result.refusal_rate_ci_low is None
    assert result.refusal_rate_ci_high is None
    assert result.ambiguous == 3


def test_refusal_rate_by_category_split_correctly() -> None:
    """When ``prompts`` is supplied, the aggregator should compute a
    per-category refusal rate. We construct a synthetic suite with one of
    each category and confirm the breakdown matches.
    """
    prompts = [
        Prompt(id="h1", suite="s", text="t", category="harmful"),
        Prompt(id="h2", suite="s", text="t", category="harmful"),
        Prompt(id="p1", suite="s", text="t", category="helpful"),
        Prompt(id="b1", suite="s", text="t", category="borderline"),
    ]
    responses = [
        RawResponse(
            prompt_id=p.id,
            model="m",
            output="",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        )
        for p in prompts
    ]
    labels = [
        RefusalLabel(prompt_id="h1", model="m", refused=True, confidence=1.0, judge="rule"),
        RefusalLabel(prompt_id="h2", model="m", refused=False, confidence=0.0, judge="rule"),
        RefusalLabel(prompt_id="p1", model="m", refused=True, confidence=1.0, judge="rule"),
        RefusalLabel(prompt_id="b1", model="m", refused=False, confidence=0.0, judge="rule"),
    ]
    result = aggregate_results(
        model="m",
        suite="s",
        responses=responses,
        labels=labels,
        prompts=prompts,
    )
    assert result.refusal_rate_by_category["harmful"] == 0.5
    assert result.refusal_rate_by_category["helpful"] == 1.0
    assert result.refusal_rate_by_category["borderline"] == 0.0


def test_run_eval_respects_max_concurrent() -> None:
    """A client with a known concurrency cap must never see more in-flight
    requests than ``RunConfig.max_concurrent`` allows.
    """

    class CountingClient:
        name = "counter"

        def __init__(self) -> None:
            self.in_flight = 0
            self.peak = 0
            self._lock = asyncio.Lock()

        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            max_tokens: int,
            seed: int,
        ) -> RawResponse:
            async with self._lock:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
            try:
                # Give other coroutines a chance to start.
                await asyncio.sleep(0.005)
            finally:
                async with self._lock:
                    self.in_flight -= 1
            return RawResponse(
                prompt_id="",
                model=self.name,
                output="ok",
                generation_seconds=0.005,
                timestamp=int(time.time()),
                seed=seed,
            )

    client = CountingClient()
    config = RunConfig(
        model="counter",
        suites=["harmful_helpful"],
        seed=1,
        max_concurrent=3,
    )
    responses = _run(run_eval(client, "harmful_helpful", config))
    assert len(responses) == 15
    assert client.peak <= 3, f"peak in-flight was {client.peak}; limit was 3"


def test_judge_responses_does_not_recurse_into_asyncio_run() -> None:
    """Regression for F-R2-P0-1.

    Before the fix, ``judge_responses`` called ``asyncio.run`` internally,
    which raised ``RuntimeError`` when invoked from inside a coroutine
    already running under ``asyncio.run`` (the CLI's ``_run_async`` path
    with ``--judge llm``). The fix splits the API into an async-first
    ``ajudge_responses`` plus a sync wrapper; the CLI must use the async
    one.

    We exercise the regression by driving the full async path with an
    in-process MockTransport-backed AnthropicClient as the judge.
    """
    import httpx

    from lre.cli import _run_async
    from lre.models.anthropic_api import AnthropicClient

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "content": [{"type": "text", "text": "REFUSED | sanity"}],
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]
    try:
        client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=1)
        judge_client = AnthropicClient(
            model="claude-test", api_key="fake-key", base_url="https://example.test"
        )
        llm_judge = LLMJudge(client=judge_client, seed=0)

        async def go() -> None:
            # _run_async now returns (results, raw_responses); we only need
            # to assert no RuntimeError escapes.
            await _run_async(
                client,
                ["harmful_helpful"],
                seed=1,
                temperature=0.0,
                max_tokens=64,
                judge="llm",
                max_concurrent=4,
                llm_judge=llm_judge,
            )

        # The bug raised RuntimeError("asyncio.run() cannot be called from
        # a running event loop") — assert that no RuntimeError escapes.
        asyncio.run(go())
    finally:
        httpx.AsyncClient = real  # type: ignore[misc]


def test_ajudge_responses_with_rule_judge_returns_labels() -> None:
    """Async entry point should work even without an LLMJudge."""
    responses = [
        RawResponse(
            prompt_id=f"p{i}",
            model="m",
            output="I cannot help with that.",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        )
        for i in range(3)
    ]
    labels = asyncio.run(ajudge_responses(responses))
    assert len(labels) == 3
    assert all(lbl.refused for lbl in labels)


def test_per_category_records_none_for_all_errored_category() -> None:
    """F-R2-P2-15: when all prompts in a category errored, the breakdown
    should still register that category — with value ``None`` — rather
    than silently dropping it.
    """
    prompts = [
        Prompt(id="h1", suite="s", text="t1", category="harmful"),
        Prompt(id="p1", suite="s", text="t2", category="helpful"),
        Prompt(id="p2", suite="s", text="t3", category="helpful"),
    ]
    responses = [
        RawResponse(
            prompt_id="h1",
            model="m",
            output="I cannot help.",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        ),
        # Both helpful prompts errored.
        RawResponse(
            prompt_id="p1",
            model="m",
            output="",
            generation_seconds=-1.0,
            timestamp=0,
            seed=0,
        ),
        RawResponse(
            prompt_id="p2",
            model="m",
            output="",
            generation_seconds=-1.0,
            timestamp=0,
            seed=0,
        ),
    ]
    labels = [
        RefusalLabel(prompt_id="h1", model="m", refused=True, confidence=1.0, judge="rule"),
        RefusalLabel(prompt_id="p1", model="m", refused=False, confidence=0.0, judge="rule"),
        RefusalLabel(prompt_id="p2", model="m", refused=False, confidence=0.0, judge="rule"),
    ]
    result = aggregate_results(
        model="m",
        suite="s",
        responses=responses,
        labels=labels,
        prompts=prompts,
    )
    assert result.refusal_rate_by_category["harmful"] == 1.0
    assert "helpful" in result.refusal_rate_by_category
    assert result.refusal_rate_by_category["helpful"] is None


def test_judge_responses_sync_wrapper_still_works() -> None:
    """Sync wrapper must round-trip a small batch via the rule judge."""
    responses = [
        RawResponse(
            prompt_id="p1",
            model="m",
            output="I cannot help with that.",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        ),
    ]
    labels = judge_responses(responses)
    assert len(labels) == 1
    assert labels[0].refused is True


def test_run_eval_preserves_input_order() -> None:
    """Even when fan-out is asynchronous, the runner must return responses
    in prompt-suite order. We assert by zipping ``responses`` against the
    suite's prompt ids.
    """
    client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=1)
    config = RunConfig(
        model="fake-1b",
        suites=["harmful_helpful"],
        seed=1,
        max_concurrent=8,
    )
    prompts = load_suite("harmful_helpful")
    responses = _run(run_eval(client, "harmful_helpful", config))
    assert [r.prompt_id for r in responses] == [p.id for p in prompts]


# ---------------------------------------------------------------------------
# v0.7 concurrency / fail-fast contract
# ---------------------------------------------------------------------------


def test_ajudge_responses_runs_in_parallel_under_semaphore() -> None:
    """The async judge fan-out must complete in parallel, not serial.

    Drives a fake async judge that sleeps 100ms per call. With 8 inputs
    and ``max_concurrent=4``, two waves of 4 should fit inside ~250ms;
    the previous serial implementation took 800ms+.
    """

    class _SleepyJudge:
        kind: str = "custom"

        def label(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
            raise NotImplementedError("async-only judge")

        async def alabel(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
            await asyncio.sleep(0.1)
            return RefusalLabel(
                prompt_id=prompt_id,
                model=model,
                refused=True,
                confidence=1.0,
                judge="rule",
            )

    responses = [
        RawResponse(
            prompt_id=f"p{i}",
            model="m",
            output="out",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        )
        for i in range(8)
    ]
    start = time.perf_counter()
    labels = _run(ajudge_responses(responses, judge=_SleepyJudge(), max_concurrent=4))
    elapsed = time.perf_counter() - start
    assert len(labels) == 8
    # Two waves of 4 at 100ms each = ~200ms; allow generous slack so a
    # noisy CI runner does not flake the test.
    assert elapsed < 0.4, f"ajudge_responses did not parallelise: {elapsed:.3f}s"


def test_run_eval_cancels_remaining_tasks_on_auth_error() -> None:
    """Auth errors must short-circuit the entire batch, not retry-storm.

    The previous gather-everything-and-then-raise behaviour spent the
    entire suite's retry budget before surfacing the credential bug.
    """
    from lre.models.openai_api import AuthenticationError

    seen_calls: list[int] = []

    class _AuthFailingClient:
        name = "auth-fail"

        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            max_tokens: int,
            seed: int,
        ) -> RawResponse:
            seen_calls.append(len(seen_calls))
            # The third call 401s; everything else hangs for 5 seconds.
            if len(seen_calls) == 3:
                raise AuthenticationError("simulated 401")
            await asyncio.sleep(5.0)
            return RawResponse(
                prompt_id="",
                model=self.name,
                output="ok",
                generation_seconds=0.0,
                timestamp=0,
                seed=seed,
            )

    prompts = [
        Prompt(id=f"p{i}", suite="demo", text=f"hi-{i}", category="helpful") for i in range(100)
    ]
    config = RunConfig(
        model="auth-fail",
        suites=["demo"],
        seed=0,
        max_concurrent=8,
    )
    start = time.perf_counter()
    with pytest.raises(AuthenticationError):
        _run(run_eval(_AuthFailingClient(), "demo", config, prompts=prompts))
    elapsed = time.perf_counter() - start
    # The whole batch should fail in well under a second. The pre-fix
    # behaviour finished only after the hanging tasks did.
    assert elapsed < 2.0, f"auth-error fast-fail regressed: {elapsed:.3f}s"
