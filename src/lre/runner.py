"""Async eval orchestrator.

The runner iterates prompts, fans out generation calls under an
``asyncio.Semaphore`` (so we never exceed ``RunConfig.max_concurrent`` in
flight), and records either a real :class:`RawResponse` or an
error-sentinel response. Output order is preserved.

Aggregation lives in :func:`aggregate_results`; it consumes raw responses
plus refusal labels and produces an :class:`EvalResult` carrying headline
rate, per-category breakdown, and a 95% Wilson confidence interval.
Splitting generation from judging means re-judging a cached run (e.g.
swapping in an LLM judge after the fact) does not require regenerating
responses.

The judging API exposes both an async-first entry point
(:func:`ajudge_responses`) and a sync wrapper (:func:`judge_responses`).
The async version is the canonical one for the runner: calling
``asyncio.run`` from inside a coroutine raises ``RuntimeError`` in
CPython, so the CLI ``--judge llm`` path **must** await
``ajudge_responses`` directly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Iterable, Sequence
from typing import Literal

from lre.cache import ResponseCache
from lre.judge import Judge, LLMJudge, RuleBasedJudge
from lre.models.anthropic_api import AuthenticationError as AnthropicAuthError
from lre.models.base import ModelClient
from lre.models.openai_api import AuthenticationError as OpenAIAuthError
from lre.prompts import load_suite
from lre.provenance import Provenance, collect_provenance
from lre.state import EvalResult, Prompt, PromptCategory, RawResponse, RefusalLabel, RunConfig
from lre.stats import compute_wilson_ci

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


_AUTH_ERROR_TYPES: tuple[type[BaseException], ...] = (OpenAIAuthError, AnthropicAuthError)


async def _gather_with_auth_fastfail(
    tasks: list[asyncio.Task[RawResponse]],
) -> list[RawResponse | BaseException]:
    """Wait for ``tasks`` and cancel the rest on first auth failure.

    The default ``asyncio.gather(..., return_exceptions=True)`` waits
    for every task to complete — so a 100-prompt run with a bad API key
    spends 100 retry-storms (or however the adapter's exponential
    backoff is shaped) before the auth error surfaces. This helper
    cancels remaining tasks as soon as the first auth error appears,
    so the user sees the error in well under a second.

    Other exceptions propagate normally into ``return_exceptions``
    semantics: each finished task is paired with its result (or the
    raised exception) and returned in input order.
    """
    pending = set(tasks)
    auth_exc: BaseException | None = None
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            if finished.cancelled():
                continue
            exc = finished.exception()
            if isinstance(exc, _AUTH_ERROR_TYPES):
                auth_exc = exc
                break
        if auth_exc is not None:
            break
    if auth_exc is not None:
        for task in pending:
            task.cancel()
        # Wait for the cancellations to settle so we do not leak tasks.
        # Both ``CancelledError`` and any user exception raised by a
        # still-running task are swallowed — the only error we want to
        # surface is the original auth failure.
        await asyncio.gather(*pending, return_exceptions=True)
        raise auth_exc
    # No auth error — collect results in input order. Any other
    # exception is returned alongside its task so the caller's
    # per-task sentinel logic can run unchanged.
    results: list[RawResponse | BaseException] = []
    for task in tasks:
        exc = task.exception()
        if exc is not None:
            results.append(exc)
        else:
            results.append(task.result())
    return results


async def run_eval(
    client: ModelClient,
    suite: str,
    config: RunConfig,
    *,
    cache: ResponseCache | None = None,
    sample_n: int | None = None,
    prompts: Sequence[Prompt] | None = None,
) -> list[RawResponse]:
    """Run a single suite end-to-end and return raw responses.

    The function is resilient to per-prompt failures: if a generation call
    raises, the runner logs the exception and emits a sentinel
    :class:`RawResponse` with ``output=""`` and ``generation_seconds=-1.0``
    so downstream stats still have a row for that prompt.

    Concurrency is capped at ``config.max_concurrent`` via an
    ``asyncio.Semaphore``; output order matches input order.

    Parameters
    ----------
    cache:
        Optional :class:`~lre.cache.ResponseCache`. When supplied, the
        runner consults the cache before calling ``client.generate``
        and writes successful generations back. The cache key includes
        ``(model, prompt, seed, temperature, max_tokens)``.
    sample_n:
        When set, sample this many prompts deterministically (using
        ``config.seed``) instead of running the full suite. If
        ``sample_n`` exceeds the suite size, the runner logs a warning
        and uses all prompts.
    prompts:
        Override the prompts loaded from the suite name — used by
        callers that already have the prompt list in hand (e.g. the
        CLI, which needs the same list to label categories).
    """
    if prompts is None:
        prompts = load_suite(suite)
    if sample_n is not None and sample_n > 0:
        prompts = _deterministic_sample(prompts, sample_n, config.seed)
    logger.info(
        "starting eval suite=%s model=%s prompts=%d seed=%d max_concurrent=%d",
        suite,
        client.name,
        len(prompts),
        config.seed,
        config.max_concurrent,
    )
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _bounded(prompt: Prompt) -> RawResponse:
        async with semaphore:
            return await _generate_one(client, prompt, config, cache=cache)

    tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
    # Auth errors are non-retryable. Use ``_gather_with_auth_fastfail``
    # so the first 401 / 403 cancels the remaining tasks instead of
    # waiting for the entire suite to retry-storm to completion. Other
    # exceptions still flow through to the per-task sentinel path below.
    raw_results = await _gather_with_auth_fastfail(tasks)
    responses: list[RawResponse] = []
    for index, (prompt, result) in enumerate(zip(prompts, raw_results, strict=True), start=1):
        if isinstance(result, BaseException):
            # _generate_one is supposed to catch its own exceptions; if one
            # escapes, treat it as an error sentinel so we never poison the
            # output list.
            logger.exception("task for prompt_id=%s raised %r", prompt.id, result)
            responses.append(
                RawResponse(
                    prompt_id=prompt.id,
                    model=client.name,
                    output="",
                    generation_seconds=-1.0,
                    timestamp=int(time.time()),
                    seed=config.seed,
                )
            )
        else:
            responses.append(result)
            logger.debug(
                "generated %d/%d prompt_id=%s latency=%.3fs",
                index,
                len(prompts),
                prompt.id,
                result.generation_seconds,
            )
    return responses


async def _generate_one(
    client: ModelClient,
    prompt: Prompt,
    config: RunConfig,
    *,
    cache: ResponseCache | None = None,
) -> RawResponse:
    if cache is not None:
        cached = cache.get(
            model=client.name,
            prompt=prompt.text,
            seed=config.seed,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        if cached is not None:
            # Stamp the prompt_id (cache may carry the original) and
            # mark this as a zero-latency replay so downstream
            # consumers can tell hits apart from real generations.
            return cached.model_copy(update={"prompt_id": prompt.id, "generation_seconds": 0.0})
    try:
        raw = await client.generate(
            prompt.text,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    except (OpenAIAuthError, AnthropicAuthError):
        # Auth errors are non-retryable and per-prompt logging
        # would spam stderr for every remaining prompt. Re-raise so the
        # outer ``asyncio.gather`` cancels siblings and the CLI surfaces
        # a single clean error.
        raise
    except Exception:
        logger.exception("generation failed for prompt_id=%s", prompt.id)
        return RawResponse(
            prompt_id=prompt.id,
            model=client.name,
            output="",
            generation_seconds=-1.0,
            timestamp=int(time.time()),
            seed=config.seed,
        )
    # Adapters typically leave prompt_id blank; the runner is the canonical
    # place to stamp it so downstream code always sees a populated value.
    response = raw.model_copy(update={"prompt_id": prompt.id})
    if cache is not None and response.generation_seconds >= 0:
        # Only cache successful generations — error sentinels with
        # ``generation_seconds == -1`` would otherwise poison the cache.
        cache.put(
            response,
            prompt=prompt.text,
            seed=config.seed,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    return response


def _deterministic_sample(prompts: Sequence[Prompt], n: int, seed: int) -> list[Prompt]:
    """Sample ``n`` prompts deterministically using ``seed``.

    If ``n`` exceeds the available prompt count, the runner logs a
    warning and returns the full set. The output preserves the
    original prompt order so downstream artefacts (raw-response JSONL,
    per-category breakdowns) remain reproducible.
    """
    available = list(prompts)
    if n >= len(available):
        if n > len(available):
            logger.warning(
                "sample_n=%d exceeds suite size %d; using all prompts",
                n,
                len(available),
            )
        return available
    rng = random.Random(seed)
    sampled = rng.sample(available, n)
    # Reorder to follow the original suite ordering so the output is
    # easier to compare against the full-suite run.
    chosen_ids = {p.id for p in sampled}
    return [p for p in available if p.id in chosen_ids]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def _resolve_judge(
    *,
    judge: Judge | RuleBasedJudge | LLMJudge | None,
    kind: str,
    llm_judge: LLMJudge | None,
) -> Judge:
    """Disambiguate ``(judge, kind, llm_judge)`` triple into a concrete Judge.

    Shared helper so the sync and async judging entry points stay in lockstep.

    When ``judge`` is supplied, the helper enforces ``judge.kind`` at
    runtime: the static :class:`~lre.judge.JudgeKindLiteral` cannot
    constrain third-party custom judges, so we validate the value here
    instead of letting downstream consumers swallow a stray label.
    """
    if judge is not None:
        # Runtime validation of the Judge protocol's ``kind`` attribute.
        # The Literal annotation is statically checked but third-party
        # judges authored without mypy could still set kind='something
        # else'. Reject early with a clear message. The canonical set
        # lives in :data:`lre.state.JudgeKind` — keep this tuple in
        # lockstep with that declaration.
        kind_attr = getattr(judge, "kind", None)
        if kind_attr not in ("rule", "llm", "manual", "custom"):
            msg = (
                f"Judge.kind must be one of 'rule', 'llm', 'manual', 'custom'; "
                f"got {kind_attr!r}. Third-party judges should declare "
                "kind='custom'."
            )
            raise ValueError(msg)
        return judge
    if kind == "rule":
        return RuleBasedJudge()
    if kind == "llm":
        if llm_judge is None:
            msg = (
                "LLM-judge requires passing an `llm_judge` instance into "
                "judge_responses(kind='llm', llm_judge=...). The rule judge "
                "is the default when no llm_judge is configured."
            )
            raise ValueError(msg)
        return llm_judge
    msg = f"unknown judge kind {kind!r}; expected 'rule' or 'llm'"
    raise ValueError(msg)


async def ajudge_responses(
    responses: Sequence[RawResponse],
    *,
    judge: Judge | RuleBasedJudge | LLMJudge | None = None,
    kind: Literal["rule", "llm"] = "rule",
    llm_judge: LLMJudge | None = None,
    max_concurrent: int = 4,
) -> list[RefusalLabel]:
    """Async-first variant of :func:`judge_responses`.

    Always available — even when the rule judge is selected, this entry
    point exists so callers driving the runner from inside an event loop
    do not need to spawn a nested ``asyncio.run`` (which raises in CPython).

    Labels are computed concurrently under an ``asyncio.Semaphore``
    capped at ``max_concurrent`` so an LLM-backed judge fans out
    instead of serialising one network round-trip at a time. The rule
    judge does no I/O, so the parallelism is effectively a no-op for it
    — same correctness, same byte-stable order.

    Parameters
    ----------
    responses:
        Raw responses to label.
    judge:
        Optional pre-built judge instance — must implement the
        :class:`~lre.judge.Judge` protocol. When supplied, ``kind`` is
        ignored.
    kind:
        ``"rule"`` (default) instantiates a :class:`RuleBasedJudge`.
        ``"llm"`` requires ``llm_judge`` to be supplied; raises
        ``ValueError`` otherwise.
    llm_judge:
        Pre-wired :class:`LLMJudge`. Required when ``kind="llm"`` and
        ``judge`` is not supplied.
    max_concurrent:
        Cap on judge calls in flight. Mirrors the generation runner's
        own concurrency knob — 4 by default.
    """
    active_judge = _resolve_judge(judge=judge, kind=kind, llm_judge=llm_judge)
    semaphore = asyncio.Semaphore(max(max_concurrent, 1))

    async def _bounded(r: RawResponse) -> RefusalLabel:
        async with semaphore:
            return await active_judge.alabel(r.prompt_id, r.model, r.output)

    # asyncio.gather preserves input order regardless of completion order.
    return list(await asyncio.gather(*(_bounded(r) for r in responses)))


def judge_responses(
    responses: Sequence[RawResponse],
    *,
    judge: Judge | RuleBasedJudge | LLMJudge | None = None,
    kind: Literal["rule", "llm"] = "rule",
    llm_judge: LLMJudge | None = None,
) -> list[RefusalLabel]:
    """Apply a judge to a batch of responses (sync wrapper).

    This is the legacy synchronous entry point. Internally it dispatches
    to :func:`ajudge_responses` via ``asyncio.run`` — so it **cannot** be
    called from within a running event loop. Async callers (the CLI's
    ``_run_async`` path, the runner's internal use) should call
    :func:`ajudge_responses` directly.

    Parameters
    ----------
    responses:
        Raw responses to label.
    judge:
        Optional pre-built judge instance. If supplied, ``kind`` is ignored
        — the caller has already chosen.
    kind:
        ``"rule"`` (default) instantiates a :class:`RuleBasedJudge`.
        ``"llm"`` requires ``llm_judge`` to be supplied as a real
        :class:`LLMJudge` instance and raises ``ValueError`` otherwise.
    llm_judge:
        Pre-wired :class:`LLMJudge`. Required when ``kind="llm"`` and
        ``judge`` is not supplied.
    """
    return asyncio.run(ajudge_responses(responses, judge=judge, kind=kind, llm_judge=llm_judge))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    *,
    model: str,
    suite: str,
    responses: Sequence[RawResponse],
    labels: Sequence[RefusalLabel],
    prompts: Sequence[Prompt] | None = None,
    with_provenance: bool = False,
    seed: int | None = None,
    provenance: Provenance | None = None,
) -> EvalResult:
    """Combine raw responses + judge labels into a single :class:`EvalResult`.

    Parameters
    ----------
    prompts:
        Optional prompt metadata. When supplied, the resulting
        :attr:`EvalResult.refusal_rate_by_category` is populated.
        Categories whose prompts *all* errored still register in the
        dict with ``None`` so downstream consumers can distinguish "no
        data" from "missing category".
    with_provenance:
        When True, attach a :class:`~lre.provenance.Provenance` snapshot
        to the result. ``lre run`` enables this; ``lre demo`` leaves it
        off so the demo path stays byte-stable across reruns (the
        timestamp embedded in provenance would otherwise drift each
        run). Either ``seed`` or a pre-built ``provenance`` must be
        supplied when this is True.
    seed:
        Run seed forwarded into the provenance snapshot when one is
        collected here.
    provenance:
        Pre-built provenance snapshot — useful when several
        ``EvalResult`` rows share the same run-level metadata so we
        only do the work once.
    """
    if len(responses) != len(labels):
        msg = f"responses ({len(responses)}) and labels ({len(labels)}) length mismatch"
        raise ValueError(msg)
    if prompts is not None and len(prompts) != len(responses):
        msg = f"prompts ({len(prompts)}) and responses ({len(responses)}) length mismatch"
        raise ValueError(msg)

    total = len(responses)
    refused = 0
    complied = 0
    ambiguous = 0
    per_category: dict[PromptCategory, list[bool]] = {}
    seen_categories: set[PromptCategory] = set()
    for index, (resp, lbl) in enumerate(zip(responses, labels, strict=True)):
        if prompts is not None:
            # Track every category we have at least one prompt for, so the
            # final breakdown can report ``None`` for all-errored categories
            # rather than silently dropping them.
            seen_categories.add(prompts[index].category)
        if resp.generation_seconds < 0:
            # Errored generations are not counted as either refusal or
            # compliance — they are flagged as ambiguous so that they still
            # appear in the totals but do not bias the headline rate.
            ambiguous += 1
            continue
        if lbl.refused:
            refused += 1
        else:
            complied += 1
        if prompts is not None:
            cat: PromptCategory = prompts[index].category
            per_category.setdefault(cat, []).append(lbl.refused)

    refusal_rate: float | None
    ci_low: float | None
    ci_high: float | None
    judged = refused + complied
    if judged > 0:
        refusal_rate = refused / judged
        ci_low, ci_high = compute_wilson_ci(refused, judged, confidence=0.95)
    else:
        refusal_rate = None
        ci_low = None
        ci_high = None

    by_category: dict[PromptCategory, float | None] = {}
    if prompts is not None:
        for cat in seen_categories:
            bools = per_category.get(cat, [])
            by_category[cat] = (sum(bools) / len(bools)) if bools else None

    # v0.8 (P0-5): both percentiles use the same nearest-rank definition
    # so they cannot silently drift apart. Pre-v0.8 used
    # ``statistics.median`` (linear interpolation, R Type 7) for p50 and
    # nearest-rank for p99 — a reported "p50 < p99" relationship that
    # held by accident on continuous data but became inconsistent on
    # small or repeated samples.
    latencies = [r.generation_seconds for r in responses if r.generation_seconds >= 0]
    p50 = _percentile(latencies, 0.50) if latencies else 0.0
    p99 = _percentile(latencies, 0.99) if latencies else 0.0

    resolved_provenance: Provenance | None
    if provenance is not None:
        resolved_provenance = provenance
    elif with_provenance:
        if seed is None:
            msg = (
                "aggregate_results(with_provenance=True) requires either a "
                "`seed` int or a pre-built `provenance` snapshot."
            )
            raise ValueError(msg)
        resolved_provenance = collect_provenance(seed)
    else:
        resolved_provenance = None

    return EvalResult(
        model=model,
        suite=suite,
        total=total,
        refused=refused,
        complied=complied,
        ambiguous=ambiguous,
        refusal_rate=refusal_rate,
        refusal_rate_ci_low=ci_low,
        refusal_rate_ci_high=ci_high,
        refusal_rate_by_category=by_category,
        latency_p50_s=round(p50, 6),
        latency_p99_s=round(p99, 6),
        provenance=resolved_provenance,
    )


def _percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile; deterministic, no numpy needed.

    Uses Hyndman & Fan Type 1 (the inverted-CDF rule, matching
    ``numpy.percentile(..., method='inverted_cdf')``): the value at rank
    ``ceil(q * n)`` of the sorted input, 1-indexed. No interpolation is
    performed — the result is always one of the input values. ``math.ceil``
    (not ``round``) is used so that ``q=0.025`` over a 20-element list
    yields the 1st element (not the 0th via banker's rounding) and
    ``q=0.99`` over a 150-element list yields the 149th (not the 150th).

    Note: ``method='lower'`` in numpy uses ``floor(q * (n - 1))`` (a
    different convention). The implementation here is the inverted-CDF
    rule, which is the rank that NIST and most percentile tables call
    "Type 1" — pinning it makes the latency contract auditable.

    The runner uses this for BOTH p50 and p99 — pre-v0.8, p50 used
    ``statistics.median`` (linear-interpolation, R Type 7) which produced
    a value not present in the input set for even-sized samples. Pinning
    both percentiles to the same definition makes the latency contract
    auditable.
    """
    items = sorted(values)
    if not items:
        return 0.0
    if q <= 0:
        return items[0]
    if q >= 1:
        return items[-1]
    rank = max(math.ceil(q * len(items)), 1)
    rank = min(rank, len(items))
    return items[rank - 1]
