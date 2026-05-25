"""Tests for the deterministic :class:`FakeModelClient`.

The synthetic client is exercised indirectly by the runner tests; this
file pins the contract directly so a regression in the hashing or seed-
mixing path is caught without depending on the full runner pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lre.synthetic import _COMPLIANCE_TEMPLATE, _REFUSAL_TEMPLATE, FakeModelClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_fake_client_generate_is_deterministic_across_reruns() -> None:
    """Two independent clients constructed identically must emit the
    exact same output for the same prompt + seed.
    """
    client_a = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=11)
    client_b = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=11)
    out_a = _run(client_a.generate("hello", temperature=0.0, max_tokens=64, seed=7))
    out_b = _run(client_b.generate("hello", temperature=0.0, max_tokens=64, seed=7))
    assert out_a.output == out_b.output


def test_fake_client_rejects_out_of_range_refusal_rate() -> None:
    with pytest.raises(ValueError, match="refusal_rate"):
        FakeModelClient(name="fake-1b", refusal_rate=1.5)
    with pytest.raises(ValueError, match="refusal_rate"):
        FakeModelClient(name="fake-1b", refusal_rate=-0.1)


def test_fake_client_uses_configured_latency() -> None:
    client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=1, latency_seconds=0.05)
    raw = _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    assert raw.generation_seconds == 0.05


def test_fake_client_seed_mixing_changes_output_distribution() -> None:
    """The per-call seed is mixed into the per-client seed; two distinct
    seeds at the same client should produce different outputs on average.
    """
    client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=42)
    differing = 0
    for seed in range(20):
        a = _run(client.generate("same prompt", temperature=0.0, max_tokens=64, seed=seed))
        b = _run(client.generate("same prompt", temperature=0.0, max_tokens=64, seed=seed + 1000))
        if a.output != b.output:
            differing += 1
    # With 20 trials at refusal_rate=0.5, we expect about half to differ.
    # Anything from 4..16 is a comfortably loose check.
    assert 4 <= differing <= 16, f"seed mixing looks broken: {differing}/20 differed"


def test_fake_client_refusal_rate_zero_always_complies() -> None:
    client = FakeModelClient(name="fake-1b", refusal_rate=0.0, seed=42)
    for i in range(20):
        raw = _run(client.generate(f"p{i}", temperature=0.0, max_tokens=64, seed=i))
        assert raw.output == _COMPLIANCE_TEMPLATE


def test_fake_client_refusal_rate_one_always_refuses() -> None:
    client = FakeModelClient(name="fake-1b", refusal_rate=1.0, seed=42)
    for i in range(20):
        raw = _run(client.generate(f"p{i}", temperature=0.0, max_tokens=64, seed=i))
        assert raw.output == _REFUSAL_TEMPLATE


def test_fake_client_different_prompts_can_get_different_outputs() -> None:
    """Two prompts with the same id but different text should hash to
    different refusal scores (the hash uses the prompt text, not the id).
    """
    client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=1)
    differing = 0
    for i in range(40):
        a = _run(client.generate(f"prompt-A-{i}", temperature=0.0, max_tokens=64, seed=0))
        b = _run(client.generate(f"prompt-B-{i}", temperature=0.0, max_tokens=64, seed=0))
        if a.output != b.output:
            differing += 1
    assert 8 <= differing <= 32, f"prompt-text mixing looks broken: {differing}/40"
