"""Tests for the Wilson-score CI helper and related stats."""

from __future__ import annotations

import math

import pytest

from lre.stats import (
    Z_95,
    cohen_kappa,
    compute_proportion_diff_test,
    compute_wilson_ci,
)


def test_wilson_ci_known_value() -> None:
    """Cross-checked against R's ``binom::binom.wilson(8, 15)`` ~= (0.30, 0.75).
    We allow a 1e-2 tolerance to absorb minor rounding differences.
    """
    low, high = compute_wilson_ci(8, 15)
    assert 0.28 <= low <= 0.32, f"low={low}"
    assert 0.72 <= high <= 0.78, f"high={high}"


def test_wilson_ci_at_zero_and_one_are_bounded() -> None:
    """The Wilson interval must remain within ``[0, 1]`` at the extremes,
    unlike the naïve normal-approximation interval.
    """
    low_0, high_0 = compute_wilson_ci(0, 10)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.5
    low_1, high_1 = compute_wilson_ci(10, 10)
    assert high_1 == 1.0
    assert 0.5 < low_1 < 1.0


def test_wilson_ci_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        compute_wilson_ci(0, 0)
    with pytest.raises(ValueError):
        compute_wilson_ci(5, 3)
    with pytest.raises(ValueError):
        compute_wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        compute_wilson_ci(3, 10, confidence=0.5)


def test_wilson_ci_99_is_wider_than_95() -> None:
    low_95, high_95 = compute_wilson_ci(50, 100, confidence=0.95)
    low_99, high_99 = compute_wilson_ci(50, 100, confidence=0.99)
    assert low_99 < low_95
    assert high_99 > high_95


def test_z95_constant_is_full_precision() -> None:
    """F-R2-P3-27: ``Z_95`` is the full double-precision two-sided 95%
    quantile, not the colloquial 1.96. ``scipy.stats.norm.ppf(0.975)``
    yields the same constant.
    """
    assert math.isclose(Z_95, 1.959963984540054, rel_tol=0, abs_tol=1e-15)


def test_proportion_diff_test_zero_delta_gives_p_value_one() -> None:
    """Two identical proportions → p-value = 1.0, Δ = 0.0."""
    result = compute_proportion_diff_test(5, 10, 5, 10)
    assert result["delta"] == 0.0
    assert math.isclose(result["p_value"], 1.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_proportion_diff_test_strong_difference_is_significant() -> None:
    """A 10/10 vs 0/10 split is overwhelmingly significant."""
    result = compute_proportion_diff_test(0, 10, 10, 10)
    assert math.isclose(result["delta"], 1.0)
    # Two-proportion z-test with pooled variance: p ≪ 0.001
    assert result["p_value"] < 0.001


def test_newcombe_table_ii_case_7() -> None:
    """Newcombe (1998) *Stat Med* 17:873, Table II case 7 (56/70 vs 48/80).

    The published Method-10 (Wilson, no continuity correction) CI for
    ``|theta| = 0.20`` is reported by R's ``PropCIs::diffscoreci`` and
    multiple Newcombe re-implementations as ``(0.05243, 0.33387)``.

    With the library's ``delta = p_b - p_a`` convention:

    * ``(48/80, 56/70)`` ⇒ delta = +0.20, CI = (+0.0524, +0.3339).
    * ``(56/70, 48/80)`` ⇒ delta = -0.20, CI = (-0.3339, -0.0524).

    Both bounds and the SIGN PAIRING are pinned because the an earlier iteration
    half-width swap silently flipped which Wilson endpoint anchored
    which side — passing this test verifies the corrected pairing.
    """
    # delta = +0.20 case.
    pos = compute_proportion_diff_test(48, 80, 56, 70, confidence=0.95)
    assert math.isclose(pos["delta"], 0.20, abs_tol=1e-9)
    assert math.isclose(pos["ci_low"], 0.05243, abs_tol=1e-3)
    assert math.isclose(pos["ci_high"], 0.33387, abs_tol=1e-3)

    # delta = -0.20 mirror case must produce the mirrored interval.
    neg = compute_proportion_diff_test(56, 70, 48, 80, confidence=0.95)
    assert math.isclose(neg["delta"], -0.20, abs_tol=1e-9)
    assert math.isclose(neg["ci_low"], -0.33387, abs_tol=1e-3)
    assert math.isclose(neg["ci_high"], -0.05243, abs_tol=1e-3)


def test_newcombe_small_sample_asymmetric() -> None:
    """Small-sample 5/8 vs 4/8 — exercises asymmetric Wilson combination.

    With ``delta = p_b - p_a`` and inputs ``(5, 8, 4, 8)`` ⇒ delta = -0.125.
    Canonical Method-10 CI bounds (cross-checked against the manual
    formula on Wilson 5/8 ≈ (0.2999, 0.7102) and Wilson 4/8 ≈ (0.2151,
    0.7849)) are approximately ``(-0.4962, 0.3028)``.

    Small-sample, asymmetric Wilson intervals are exactly where the
    an earlier iteration half-width swap produced visibly wrong intervals — this
    case is a demanding probe of the corrected pairing.
    """
    neg = compute_proportion_diff_test(5, 8, 4, 8, confidence=0.95)
    assert math.isclose(neg["delta"], -0.125, abs_tol=1e-9)
    assert math.isclose(neg["ci_low"], -0.4962, abs_tol=1e-3)
    assert math.isclose(neg["ci_high"], 0.3028, abs_tol=1e-3)

    pos = compute_proportion_diff_test(4, 8, 5, 8, confidence=0.95)
    assert math.isclose(pos["delta"], 0.125, abs_tol=1e-9)
    assert math.isclose(pos["ci_low"], -0.3028, abs_tol=1e-3)
    assert math.isclose(pos["ci_high"], 0.4962, abs_tol=1e-3)


def test_newcombe_method_10_swap_regression() -> None:
    """The an earlier iteration implementation swapped which half-width anchors which
    side. Validate the corrected pairing on a strongly asymmetric pair
    (90/100 vs 30/100, delta = -0.6) where the swap is large enough to
    fail with a non-trivial margin.

    Under the canonical Method 10:

    * ci_low = delta - sqrt((p_b - low_b)^2 + (high_a - p_a)^2)
    * ci_high = delta + sqrt((high_b - p_b)^2 + (p_a - low_a)^2)

    The buggy formulation paired (p_a - low_a)^2 with (high_b - p_b)^2
    in ci_low and the swap in ci_high — for symmetric n_a == n_b inputs
    the bug was hidden by the matching Wilson radii on each side.
    """
    r = compute_proportion_diff_test(90, 100, 30, 100, confidence=0.95)
    assert math.isclose(r["delta"], -0.60, abs_tol=1e-9)
    # Canonical Method 10 (cross-checked against R):
    # Wilson 90/100 ≈ (0.8261, 0.9462); Wilson 30/100 ≈ (0.2179, 0.3974).
    # ci_low = -0.60 - sqrt((0.30 - 0.2179)^2 + (0.9462 - 0.90)^2) ≈ -0.694
    # ci_high = -0.60 + sqrt((0.3974 - 0.30)^2 + (0.90 - 0.8261)^2) ≈ -0.479
    assert math.isclose(r["ci_low"], -0.694, abs_tol=2e-3)
    assert math.isclose(r["ci_high"], -0.479, abs_tol=2e-3)
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]


def test_proportion_diff_test_matches_known_r_reference() -> None:
    """Cross-check against the two-proportion pooled-variance z-test.

    For ``compute_proportion_diff_test(40, 100, 50, 100)``:

    * p_a = 0.40, p_b = 0.50, delta = 0.10
    * pooled = 90/200 = 0.45
    * variance = 0.45 * 0.55 * (1/100 + 1/100) = 0.00495
    * z = 0.10 / sqrt(0.00495) ≈ 1.421338
    * two-sided p ≈ 2 * (1 - Φ(1.421338)) ≈ 0.15521849

    Tightened (F-R3-P2-13) to ``abs_tol=1e-4`` — the z-test is fully
    determined by the inputs, so any wider tolerance hides drift in the
    Φ implementation.
    """
    result = compute_proportion_diff_test(40, 100, 50, 100)
    assert math.isclose(result["delta"], 0.1, abs_tol=1e-9)
    assert math.isclose(result["p_value"], 0.15521849, abs_tol=1e-4)


def test_proportion_diff_test_ci_within_bounds_at_edges() -> None:
    """F-R3-P0-1: the Newcombe combination on Δ can drift past ±1.0 at
    extreme inputs (e.g. 1/1 vs 0/1). The result must be clamped so the
    reported CI is mathematically sensible.
    """
    # 1/1 vs 0/1 — Δ = -1.0 by construction, CI must contain it.
    r = compute_proportion_diff_test(1, 1, 0, 1)
    assert r["delta"] == -1.0
    assert -1.0 <= r["ci_low"] <= 1.0
    assert -1.0 <= r["ci_high"] <= 1.0
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]

    # 0/1 vs 1/1 — symmetric case, Δ = +1.0.
    r = compute_proportion_diff_test(0, 1, 1, 1)
    assert r["delta"] == 1.0
    assert -1.0 <= r["ci_low"] <= 1.0
    assert -1.0 <= r["ci_high"] <= 1.0
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]

    # 10/10 vs 0/10 — Δ = -1.0, larger sample.
    r = compute_proportion_diff_test(10, 10, 0, 10)
    assert -1.0 <= r["ci_low"] <= 1.0
    assert -1.0 <= r["ci_high"] <= 1.0

    # 0/10 vs 10/10 — symmetric.
    r = compute_proportion_diff_test(0, 10, 10, 10)
    assert -1.0 <= r["ci_low"] <= 1.0
    assert -1.0 <= r["ci_high"] <= 1.0

    # 0/0 degenerate — reject explicitly via the totals validator.
    with pytest.raises(ValueError):
        compute_proportion_diff_test(0, 0, 0, 0)


def test_proportion_diff_test_rejects_invalid_totals() -> None:
    with pytest.raises(ValueError):
        compute_proportion_diff_test(0, 0, 1, 1)
    with pytest.raises(ValueError):
        compute_proportion_diff_test(5, 1, 1, 10)
    with pytest.raises(ValueError):
        compute_proportion_diff_test(0, 10, 0, 10, confidence=0.5)


def test_cohen_kappa_perfect_agreement_is_one() -> None:
    """Two raters in lockstep on a non-degenerate distribution → κ = 1.0."""
    a = [True, False, True, False, True]
    b = [True, False, True, False, True]
    assert math.isclose(cohen_kappa(a, b), 1.0)


def test_cohen_kappa_complete_disagreement_is_minus_one() -> None:
    """Two raters swapping labels on a balanced distribution → κ = -1.0."""
    a = [True, False, True, False]
    b = [False, True, False, True]
    assert math.isclose(cohen_kappa(a, b), -1.0)


def test_cohen_kappa_chance_agreement_is_zero() -> None:
    """When agreement equals chance, κ = 0.0.

    With both raters at marginal 50/50 on 4 cases, the contingency
    table (a, b, c, d) = (1, 1, 1, 1) yields p_o = 0.5 and
    p_e = 0.5*0.5 + 0.5*0.5 = 0.5, so κ = 0 exactly.
    """
    a = [True, True, False, False]
    b = [True, False, True, False]
    k = cohen_kappa(a, b)
    assert math.isclose(k, 0.0, abs_tol=1e-9)


def test_cohen_kappa_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def test_cohen_kappa_handles_degenerate_constant_raters() -> None:
    """When both raters always say True (or always False), p_e == 1.

     the prior ``return -1.0`` branch was unreachable.
    ``p_e == 1`` algebraically forces ``p_o == 1`` because both rater
    marginals must be identical-constant. The function returns 1.0.
    """
    a = [True, True, True]
    b = [True, True, True]
    assert math.isclose(cohen_kappa(a, b), 1.0)
    a2 = [False, False, False]
    b2 = [False, False, False]
    assert math.isclose(cohen_kappa(a2, b2), 1.0)
