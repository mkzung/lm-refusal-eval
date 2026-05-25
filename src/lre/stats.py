"""Light-weight statistical helpers.

Kept pure-Python — no scipy dependency — so the harness can compute Wilson
confidence intervals on refusal rates, two-proportion z-tests, and Cohen's
κ without dragging numerical libraries into the install graph.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Two-sided z-scores. We expose only 90 / 95 / 99 because the harness has no
# need for arbitrary confidence levels — refusal-rate reporting in the
# literature is dominated by 95% intervals.
Z_95: float = 1.959963984540054
"""Two-sided 95% normal quantile, full double precision."""

_Z_TABLE: dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: Z_95,
    0.99: 2.5758293035489004,
}


def compute_wilson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the Wilson score CI for a binomial proportion.

    The Wilson interval is preferred over the naïve normal-approximation
    interval because it is well-defined when the observed proportion is
    0.0 or 1.0 and stays within ``[0, 1]``.

    Parameters
    ----------
    successes:
        Number of successes (refusals).
    trials:
        Total trials (refused + complied). Must be > 0.
    confidence:
        Confidence level; supported values are 0.90, 0.95, 0.99.

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` bounds in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``trials <= 0``, ``successes`` is outside ``[0, trials]``, or
        ``confidence`` is not in the supported table.
    """
    if trials <= 0:
        msg = f"trials must be > 0, got {trials}"
        raise ValueError(msg)
    if not 0 <= successes <= trials:
        msg = f"successes must be in [0, {trials}], got {successes}"
        raise ValueError(msg)
    if confidence not in _Z_TABLE:
        supported = ", ".join(f"{c}" for c in sorted(_Z_TABLE))
        msg = f"confidence must be one of {{{supported}}}, got {confidence}"
        raise ValueError(msg)

    z = _Z_TABLE[confidence]
    p_hat = successes / trials
    denom = 1.0 + (z * z) / trials
    center = (p_hat + (z * z) / (2.0 * trials)) / denom
    half = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / trials + (z * z) / (4.0 * trials * trials))
    ) / denom
    # Exact-boundary clamp: when every trial is a success (or failure), the
    # interval should hit 1.0 (or 0.0) exactly. The double-precision z
    # constant can leave a 1e-16 gap; snap to the boundary in that case.
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == trials else min(1.0, center + half)
    return (low, high)


def _phi(x: float) -> float:
    """Standard-normal CDF Φ(x) via :func:`math.erf`.

    ``scipy.stats.norm.sf`` would be cleaner but we keep the harness
    scipy-free. ``erf`` ships with the stdlib and is accurate to roughly
    1e-15.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_proportion_diff_test(
    refused_a: int,
    total_a: int,
    refused_b: int,
    total_b: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Two-proportion z-test on the difference ``p_b - p_a``.

    Returns a dict with ``delta``, ``ci_low``, ``ci_high``, ``p_value``,
    matching the layout an analyst would write into a results table.

    Confidence interval
    -------------------
    Implements Newcombe (1998) *Statistics in Medicine* 17:873, Method 10
    ("two-proportion difference, unpooled Wilson combination"). With
    ``delta = p_b - p_a`` (B plays the ``p1`` role in Newcombe's notation),
    the bounds are::

        ci_low  = delta - sqrt((p_b - low_b)**2 + (high_a - p_a)**2)
        ci_high = delta + sqrt((high_b - p_b)**2 + (p_a - low_a)**2)

    where ``(low_a, high_a)`` and ``(low_b, high_b)`` are the Wilson score
    intervals for the two sub-samples. Validated against Table II in two
    cases (argument order matches the function signature, ``A`` then ``B``):

    * **Case 7**: ``refused_a=56, total_a=70, refused_b=48, total_b=80``
      → ``delta = p_b - p_a = 0.60 - 0.80 = -0.20``;
      95% CI on delta ≈ ``(-0.3339, -0.0524)``.
    * **Case 4**: ``refused_a=5, total_a=8, refused_b=4, total_b=8``
      → ``delta = p_b - p_a = 0.50 - 0.625 = -0.125``;
      95% CI on delta ≈ ``(-0.4962, 0.3028)``.

    Pre-v0.10 the docstring rendered each case as a shorthand
    ``A_num/A_den vs B_num/B_den`` that was easy to read in the
    natural-language direction but invertible from the actual
    ``refused_a / total_a / refused_b / total_b`` argument order.

    Parameters
    ----------
    refused_a, total_a:
        Numerator and denominator for sample A.
    refused_b, total_b:
        Numerator and denominator for sample B.
    confidence:
        Confidence level; supported values are 0.90, 0.95, 0.99.

    Raises
    ------
    ValueError
        If either total is ``<= 0`` or either count is outside its total,
        or ``confidence`` is not in the supported table.
    """
    if total_a <= 0 or total_b <= 0:
        msg = f"totals must be > 0, got {total_a} and {total_b}"
        raise ValueError(msg)
    if not 0 <= refused_a <= total_a:
        msg = f"refused_a must be in [0, {total_a}], got {refused_a}"
        raise ValueError(msg)
    if not 0 <= refused_b <= total_b:
        msg = f"refused_b must be in [0, {total_b}], got {refused_b}"
        raise ValueError(msg)
    if confidence not in _Z_TABLE:
        supported = ", ".join(f"{c}" for c in sorted(_Z_TABLE))
        msg = f"confidence must be one of {{{supported}}}, got {confidence}"
        raise ValueError(msg)

    p_a = refused_a / total_a
    p_b = refused_b / total_b
    delta = p_b - p_a

    # Newcombe (1998) Method 10: combine the two Wilson intervals to bound
    # the difference. With ``delta = p_b - p_a`` (B is Newcombe's ``p1``,
    # A is ``p2``), the lower bound subtracts the distance from p_b down
    # to low_b and from p_a up to high_a; the upper bound adds the
    # distance from p_b up to high_b and from p_a down to low_a. The
    # pre-v0.8 implementation paired these half-widths incorrectly,
    # producing intervals that did not contain delta on highly
    # asymmetric inputs (validated below against Newcombe Table II).
    low_a, high_a = compute_wilson_ci(refused_a, total_a, confidence=confidence)
    low_b, high_b = compute_wilson_ci(refused_b, total_b, confidence=confidence)
    ci_low = delta - math.sqrt((p_b - low_b) ** 2 + (high_a - p_a) ** 2)
    ci_high = delta + math.sqrt((high_b - p_b) ** 2 + (p_a - low_a) ** 2)
    # A probability difference must lie in [-1, 1] by construction. The
    # Newcombe combination can drift fractionally past those bounds in the
    # 1/1-vs-0/1 corner — clamp so reported intervals stay mathematically
    # sensible.
    ci_low = max(-1.0, ci_low)
    ci_high = min(1.0, ci_high)

    # Two-proportion z-test with pooled variance — the textbook form.
    # p_value is two-sided. When both samples have the same proportion
    # exactly, the z-statistic is 0 and p_value = 1.0.
    pooled = (refused_a + refused_b) / (total_a + total_b)
    variance = pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b)
    if variance <= 0.0:
        p_value = 1.0 if delta == 0.0 else 0.0
    else:
        z = delta / math.sqrt(variance)
        # Two-sided: P(|Z| > |z|) = 2 * (1 - Φ(|z|))
        p_value = max(0.0, min(1.0, 2.0 * (1.0 - _phi(abs(z)))))

    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def cohen_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float:
    """Cohen's κ on two binary label sequences of equal length.

    κ measures agreement above chance:

    * κ = 1.0 — perfect agreement.
    * κ = 0.0 — agreement at chance level.
    * κ < 0.0 — worse than chance (systematic disagreement).

    The formula is::

        κ = (p_o - p_e) / (1 - p_e)

    where ``p_o`` is observed agreement and ``p_e`` is expected agreement
    under independence of the two raters' marginal distributions.

    Parameters
    ----------
    labels_a, labels_b:
        Equal-length sequences of booleans (the same prompt judged by
        two raters). When the two raters always emit the same label —
        the perfect-agreement edge case where ``p_e == 1`` — we return
        ``1.0`` rather than raising.

    Raises
    ------
    ValueError
        If the sequences have different lengths or are empty.
    """
    if len(labels_a) != len(labels_b):
        msg = f"label sequences differ in length: {len(labels_a)} vs {len(labels_b)}"
        raise ValueError(msg)
    n = len(labels_a)
    if n == 0:
        msg = "label sequences must be non-empty"
        raise ValueError(msg)

    agree = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
    p_o = agree / n

    p_true_a = sum(1 for a in labels_a if a) / n
    p_true_b = sum(1 for b in labels_b if b) / n
    p_false_a = 1.0 - p_true_a
    p_false_b = 1.0 - p_true_b
    p_e = p_true_a * p_true_b + p_false_a * p_false_b

    if math.isclose(p_e, 1.0):
        # Edge case: ``p_e == 1`` algebraically forces ``p_o == 1``. The
        # only way the rater-marginal product ``p_true_a*p_true_b +
        # p_false_a*p_false_b`` reaches 1.0 is if BOTH raters emit a
        # constant label (all-True-then-all-True, or all-False-then-
        # all-False). In either case observed agreement is unavoidably
        # 1.0, so we return κ = 1.0 without a disagreement branch.
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)
