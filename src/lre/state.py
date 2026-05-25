"""Pydantic v2 state models for refusal evaluation.

All models are immutable (``frozen=True``) so that an ``EvalResult`` snapshot
can be safely passed between async tasks, serialized to disk, and compared
across runs without worrying about in-place mutation. This is deliberate:
reproducibility is the whole point of the harness.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lre.provenance import Provenance

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PromptCategory = Literal["harmful", "helpful", "borderline"]


class Prompt(BaseModel):
    """A single evaluation prompt.

    Attributes
    ----------
    id:
        Unique identifier within the bundled suite, e.g. ``"hh-001"``.
    suite:
        Name of the suite the prompt belongs to (matches the JSONL filename
        without extension, e.g. ``"harmful_helpful"``).
    text:
        The prompt body, exactly as it will be sent to the model.
    category:
        High-level intent label used to score judge calibration.
    notes:
        Optional free-form annotation (e.g. provenance, expected behaviour).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: PromptCategory
    notes: str | None = None


# ---------------------------------------------------------------------------
# Raw responses
# ---------------------------------------------------------------------------


class RawResponse(BaseModel):
    """An untouched model output paired with its provenance metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Empty string is allowed because adapters typically leave this blank
    # for the runner to fill in; the runner overwrites it via model_copy.
    prompt_id: str = ""
    model: str = Field(min_length=1)
    output: str  # may be empty if the call errored
    generation_seconds: float
    timestamp: int  # unix seconds, useful for grouping reruns
    seed: int

    @field_validator("generation_seconds")
    @classmethod
    def _allow_sentinel_or_nonneg(cls, value: float) -> float:
        # NaN / +inf are nonsensical durations — reject them up front so
        # downstream percentile / latency reporters never have to defend.
        if math.isnan(value) or math.isinf(value):
            msg = "generation_seconds must be a finite float"
            raise ValueError(msg)
        # -1.0 sentinel means the call errored; otherwise it must be >= 0.
        if value < 0 and value != -1.0:
            msg = "generation_seconds must be >= 0 or the sentinel -1.0"
            raise ValueError(msg)
        return value


# ---------------------------------------------------------------------------
# Judge labels
# ---------------------------------------------------------------------------

# Canonical Judge.kind literal — kept in lockstep with
# :data:`lre.judge.JudgeKindLiteral`. ``manual`` is the historical alias
# the legacy demo emitted; ``custom`` is the legacy third-party-judge
# vocabulary. Both must continue to validate so existing label files
# load without migration. New code should prefer ``"custom"``.
JudgeKind = Literal["rule", "llm", "manual", "custom"]


class RefusalLabel(BaseModel):
    """A judgement of whether a response refused the prompt.

    ``matched_patterns`` is a ``tuple`` so the model is genuinely
    deep-frozen — a ``list`` field with ``frozen=True`` would still allow
    in-place ``label.matched_patterns.append(...)`` mutation, which defeats
    the byte-stable-reruns invariant.

    The score field is named ``heuristic_score`` (the current implementation). An earlier iteration it was
    called ``confidence``, which suggested a calibrated probability —
    the value is NEITHER calibrated NOR a probability, just a relative
    ranking signal in ``[0, 1]``. the current implementation label files containing the legacy
    ``confidence`` key continue to load via the validation alias.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    prompt_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    refused: bool
    heuristic_score: float = Field(
        ge=0.0,
        le=1.0,
        # Accept the legacy the current implementation key ``confidence`` so old JSON / Python
        # constructors keep working. Emit the new key on serialisation.
        validation_alias="confidence",
        serialization_alias="heuristic_score",
    )
    judge: JudgeKind
    matched_patterns: tuple[str, ...] = Field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Aggregated stats for a single (model, suite) pair.

    ``refusal_rate`` is ``None`` when every generation errored (so the
    denominator ``refused + complied`` is zero). Treating that as ``0.0``
    would silently bias scaling-law plots; ``None`` forces downstream
    consumers to handle the empty-denominator case explicitly.

    ``refusal_rate_ci_low`` / ``refusal_rate_ci_high`` carry a 95% Wilson
    score interval on the refusal rate when there is at least one
    non-errored prompt; otherwise both are ``None``.

    ``refusal_rate_by_category`` breaks the refusal rate down by prompt
    category (harmful / helpful / borderline) so over-refusal (refusing
    helpful prompts) and under-refusal (complying with harmful prompts)
    are visible at a glance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    total: int = Field(ge=0)
    refused: int = Field(ge=0)
    complied: int = Field(ge=0)
    ambiguous: int = Field(ge=0)
    refusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    refusal_rate_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    refusal_rate_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)
    refusal_rate_by_category: dict[PromptCategory, float | None] = Field(default_factory=dict)
    latency_p50_s: float = Field(ge=0.0)
    latency_p99_s: float = Field(ge=0.0)
    # Optional for backward-compat with the current implementation result files that pre-date
    # the provenance footer. Populated by ``aggregate_results`` when the
    # caller passes ``with_provenance=True`` (the ``lre run`` path) and
    # left ``None`` by ``lre demo`` so the demo's byte-stability
    # invariant (which depends on output containing no timestamps) is
    # preserved unchanged.
    provenance: Provenance | None = None

    @field_validator("refusal_rate")
    @classmethod
    def _round_rate(cls, value: float | None) -> float | None:
        # Round to 6 decimals so JSON serialization is byte-stable across
        # numpy / pure-Python sums on the same input.
        return None if value is None else round(value, 6)

    @field_validator("refusal_rate_ci_low", "refusal_rate_ci_high")
    @classmethod
    def _round_ci(cls, value: float | None) -> float | None:
        return None if value is None else round(value, 6)

    @field_validator("refusal_rate_by_category")
    @classmethod
    def _round_by_category(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        return {k: (None if v is None else round(v, 6)) for k, v in value.items()}

    @model_validator(mode="after")
    def _check_total_invariant(self) -> EvalResult:
        """Assert ``total == refused + complied + ambiguous``.

        This is the bedrock invariant of the aggregator: every response
        ends up in exactly one of the three buckets. Catching the
        mismatch at validation time prevents subtle accounting bugs from
        propagating into downstream reports.
        """
        if self.total != self.refused + self.complied + self.ambiguous:
            msg = (
                f"total ({self.total}) must equal refused ({self.refused}) "
                f"+ complied ({self.complied}) + ambiguous ({self.ambiguous})"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    """Top-level user-facing knobs for a single evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    suites: list[str] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    max_tokens: int = Field(default=512, gt=0)
    seed: int = 42
    judge: Literal["rule", "llm"] = "rule"
    # Maximum number of concurrent prompts in flight. The runner uses an
    # ``asyncio.Semaphore`` to cap fan-out; output order is preserved.
    # Setting this to 1 yields strictly sequential behaviour.
    max_concurrent: int = Field(default=4, ge=1, le=256)

    @field_validator("suites")
    @classmethod
    def _reject_empty_entries(cls, value: list[str]) -> list[str]:
        # ``Field(min_length=1)`` only rejects the empty list. Without an
        # additional pass an empty-string entry sneaks through and
        # propagates into ``load_suite("")`` later, where the failure
        # mode is a confusing "suite not found" rather than the actual
        # bug — a typo or stray comma in the caller's config. the current implementation
        # (P1-13) also rejects whitespace-only entries (``" "``, ``"\t"``)
        # for the same reason.
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                msg = "RunConfig.suites entries must be non-empty, non-whitespace strings"
                raise ValueError(msg)
        return value
