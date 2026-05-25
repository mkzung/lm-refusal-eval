"""Paired-judge ("defense-in-depth") refusal-rate analysis.

Motivation
----------
A real-world deployment rarely relies on the model's own refusal
behaviour alone. Production safety stacks pair the model with an outer
classifier — often a smaller, cheaper LLM or a regex/keyword detector —
that vetoes the model's output. The system refuses whenever *either*
component flags the response.

This module ships the tooling to measure that pipeline directly:

* :class:`PairedDefense` pairs an inner judge (typically the model's
  own self-refusal signal classified by a rule judge) with an outer
  judge (typically an LLM-judge or an external classifier).
* :func:`paired_label_responses` applies the pair to a batch of
  :class:`~lre.state.RawResponse` rows. Errored responses
  (``generation_seconds == -1.0``) are excluded from judging entirely
  — they are returned as a separate count so downstream stats can
  report them without contaminating the joint refusal rate.
* :func:`aggregate_paired_results` summarises the per-prompt outcomes
  into refusal rates: inner-only, outer-only, and the joint rate. The
  denominators correctly exclude the ``ambiguous`` (errored) bucket,
  matching :func:`lre.runner.aggregate_results` semantics.

Reference
---------
Inspired by FAR.AI's research on layered LLM defenses. The most
directly-relevant published work is the "STACK" line — Howe et al.,
*STACK: Adversarial Attacks on LLM Safeguard Pipelines* (2025),
companion blog "Layered AI Defenses Have Holes" at
https://www.far.ai/news — which finds that stacking a refusal
classifier on top of a model leaves measurable vulnerabilities that an
adaptive attacker can exploit. ``lre did`` is the measurement side of
that finding: it lets a researcher quantify how much (or how little)
the outer classifier actually shifts the joint refusal rate vs. the
model-only baseline, on whatever prompt suite they care about.

See :mod:`lre`'s ``CITATION.cff`` for the canonical reference. We do
not invent author lists or arxiv IDs when in doubt — when the upstream
metadata is uncertain, prefer the conservative
``https://www.far.ai/news`` pointer over a fabricated citation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from lre.judge import Judge
from lre.state import RawResponse, RefusalLabel


def _is_error_sentinel(response: RawResponse) -> bool:
    """Return True for the runner's error-sentinel marker.

    The runner emits ``generation_seconds == -1.0`` (and typically an
    empty ``output``) when a generation call raises. Feeding such rows
    to the inner / outer judges silently registers them as compliances
    in the legacy aggregator, biasing the joint refusal rate downward.
    Callers should skip them at the input layer (this module does so in
    :func:`paired_label_responses`) and count them separately as
    ``ambiguous``.
    """
    return response.generation_seconds < 0


class PairedDefense:
    """Two-layer refusal-pipeline model.

    A response is considered *refused-by-system* if EITHER judge flags
    it. Mathematically this is the boolean OR of the two judges'
    ``refused`` outputs; downstream we report the joint rate plus the
    individual rates so the Δ from layering is visible at a glance.

    Parameters
    ----------
    inner_judge:
        The model-side judge — typically a :class:`~lre.judge.RuleBasedJudge`
        applied to the model's own output.
    outer_judge:
        The classifier-side judge — typically an
        :class:`~lre.judge.LLMJudge` or another external classifier.
    """

    def __init__(self, inner_judge: Judge, outer_judge: Judge) -> None:
        self._inner = inner_judge
        self._outer = outer_judge

    async def alabel(self, response: RawResponse) -> tuple[RefusalLabel, RefusalLabel, bool]:
        """Return ``(inner_label, outer_label, system_refused)``.

        ``system_refused`` is ``inner.refused or outer.refused`` — the
        boolean OR that defines the joint refusal rate.
        """
        inner = await self._inner.alabel(response.prompt_id, response.model, response.output)
        outer = await self._outer.alabel(response.prompt_id, response.model, response.output)
        return inner, outer, inner.refused or outer.refused


async def paired_label_responses(
    defense: PairedDefense,
    responses: Sequence[RawResponse],
    *,
    max_concurrent: int = 4,
) -> tuple[list[RefusalLabel], list[RefusalLabel], list[bool], int]:
    """Apply a :class:`PairedDefense` to a batch of responses.

    Returns four values:

    * ``inner_labels`` — labels from the inner judge, one per
      non-errored response (preserves input order).
    * ``outer_labels`` — labels from the outer judge, parallel to
      ``inner_labels``.
    * ``joint`` — per-prompt joint refusal verdicts, parallel to the
      label lists.
    * ``ambiguous`` — count of responses that were skipped because they
      are runner error sentinels. These never reach the judges, so
      ``len(inner_labels) + ambiguous == len(responses)``.

    Error sentinels (``generation_seconds == -1.0``) are filtered out
    *before* judging. Without this filter, the rule judge would happily
    label the empty ``output`` string as "not refused" and the silent
    compliance would pull the joint refusal rate downward — a
    statistical bias that hides API outages in the refusal numbers.

    Judging is fanned out under an ``asyncio.Semaphore`` capped at
    ``max_concurrent`` so an LLM-backed outer judge does not serialise
    one network round-trip per prompt.
    """
    judgable = [r for r in responses if not _is_error_sentinel(r)]
    ambiguous = len(responses) - len(judgable)
    semaphore = asyncio.Semaphore(max(max_concurrent, 1))

    async def _bounded(resp: RawResponse) -> tuple[RefusalLabel, RefusalLabel, bool]:
        async with semaphore:
            return await defense.alabel(resp)

    judged = list(await asyncio.gather(*(_bounded(r) for r in judgable)))
    inner_labels = [item[0] for item in judged]
    outer_labels = [item[1] for item in judged]
    joint = [item[2] for item in judged]
    return inner_labels, outer_labels, joint, ambiguous


def aggregate_paired_results(
    inner_labels: Sequence[RefusalLabel],
    outer_labels: Sequence[RefusalLabel],
    joint: Sequence[bool],
    ambiguous: int = 0,
) -> dict[str, float | int]:
    """Summarise the three label streams into rates.

    Returns a dict with the following keys (rates rounded to six
    decimals for byte-stable serialisation):

    * ``total`` — total number of responses considered, including the
      errored ones counted in ``ambiguous``.
    * ``ambiguous`` — count of error-sentinel responses excluded from
      the denominators. Mirrors :func:`lre.runner.aggregate_results`.
    * ``inner_refused`` / ``outer_refused`` / ``joint_refused`` — counts
      among judged (non-ambiguous) responses.
    * ``inner_refusal_rate`` / ``outer_refusal_rate`` /
      ``joint_refusal_rate`` — rates in ``[0, 1]``, computed as
      ``count / judged`` where ``judged = total - ambiguous``.
    * ``delta_vs_inner_only`` — ``joint_rate - inner_rate`` (in [-1, 1]).

    Legacy the current implementation callers also receive ``model_refusal_rate`` /
    ``classifier_refusal_rate`` / ``model_refused`` / ``classifier_refused``
    / ``delta_vs_model_only`` aliases so existing report code keeps
    working without a coordinated upgrade. New code should prefer the
    ``inner_`` / ``outer_`` names which match the function's parameter
    vocabulary.

    Raises ``ValueError`` if the three label streams disagree in length.
    """
    if len(inner_labels) != len(outer_labels) or len(inner_labels) != len(joint):
        msg = (
            f"length mismatch: inner={len(inner_labels)} outer={len(outer_labels)} "
            f"joint={len(joint)}"
        )
        raise ValueError(msg)
    judged = len(inner_labels)
    total = judged + ambiguous
    inner_count = sum(1 for x in inner_labels if x.refused)
    outer_count = sum(1 for x in outer_labels if x.refused)
    joint_count = sum(1 for x in joint if x)
    if judged == 0:
        return {
            "total": total,
            "ambiguous": ambiguous,
            "inner_refused": 0,
            "outer_refused": 0,
            "joint_refused": 0,
            "inner_refusal_rate": 0.0,
            "outer_refusal_rate": 0.0,
            "joint_refusal_rate": 0.0,
            "delta_vs_inner_only": 0.0,
            # Legacy aliases (the current implementation vocabulary).
            "model_refused": 0,
            "classifier_refused": 0,
            "model_refusal_rate": 0.0,
            "classifier_refusal_rate": 0.0,
            "delta_vs_model_only": 0.0,
        }
    inner_rate = inner_count / judged
    outer_rate = outer_count / judged
    joint_rate = joint_count / judged
    return {
        "total": total,
        "ambiguous": ambiguous,
        "inner_refused": inner_count,
        "outer_refused": outer_count,
        "joint_refused": joint_count,
        "inner_refusal_rate": round(inner_rate, 6),
        "outer_refusal_rate": round(outer_rate, 6),
        "joint_refusal_rate": round(joint_rate, 6),
        "delta_vs_inner_only": round(joint_rate - inner_rate, 6),
        # Legacy aliases (the current implementation vocabulary).
        "model_refused": inner_count,
        "classifier_refused": outer_count,
        "model_refusal_rate": round(inner_rate, 6),
        "classifier_refusal_rate": round(outer_rate, 6),
        "delta_vs_model_only": round(joint_rate - inner_rate, 6),
    }


def to_markdown_did(summary: dict[str, float | int]) -> str:
    """Render a paired-defense summary as a Markdown table.

    The output is byte-stable for a given input (no timestamps, no
    floating-point sort orders) and is the default format for
    ``lre did --format md``.
    """
    total = int(summary.get("total", 0))
    ambiguous = int(summary.get("ambiguous", 0))
    judged = total - ambiguous
    inner_refused = int(summary.get("inner_refused", summary.get("model_refused", 0)))
    outer_refused = int(summary.get("outer_refused", summary.get("classifier_refused", 0)))
    joint_refused = int(summary.get("joint_refused", 0))
    inner_rate = float(summary.get("inner_refusal_rate", summary.get("model_refusal_rate", 0.0)))
    outer_rate = float(
        summary.get("outer_refusal_rate", summary.get("classifier_refusal_rate", 0.0))
    )
    joint_rate = float(summary.get("joint_refusal_rate", 0.0))
    delta = float(summary.get("delta_vs_inner_only", summary.get("delta_vs_model_only", 0.0)))
    lines = [
        "# Defense-in-depth refusal rates",
        "",
        f"Total responses: {total} (judged: {judged}, ambiguous/errored: {ambiguous})",
        "",
        "| Layer | Refused | Rate |",
        "|---|---|---|",
        f"| Inner (model-side) | {inner_refused} | {inner_rate:.4f} |",
        f"| Outer (classifier-side) | {outer_refused} | {outer_rate:.4f} |",
        f"| Joint (inner OR outer) | {joint_refused} | {joint_rate:.4f} |",
        "",
        f"Δ joint vs inner-only: {delta:+.4f}",
        "",
    ]
    return "\n".join(lines)
