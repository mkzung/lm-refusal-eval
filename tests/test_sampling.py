"""Tests for the ``--sample N`` quick-look feature.

The sampler is the friction-reduction feature that lets researchers
iterate on prompts before committing to a full-suite run. It must be
deterministic in ``--seed`` so two researchers comparing notes always
see the same subset.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from lre.cli import main
from lre.runner import _deterministic_sample, run_eval
from lre.state import Prompt, RunConfig
from lre.synthetic import FakeModelClient


def _build_prompts(n: int) -> list[Prompt]:
    return [
        Prompt(
            id=f"p{i:03d}",
            suite="demo",
            text=f"prompt-{i}",
            category="helpful",
        )
        for i in range(n)
    ]


def test_sample_n_returns_n_prompts() -> None:
    prompts = _build_prompts(20)
    out = _deterministic_sample(prompts, 5, seed=42)
    assert len(out) == 5


def test_sample_n_is_deterministic_with_same_seed() -> None:
    prompts = _build_prompts(20)
    a = _deterministic_sample(prompts, 5, seed=42)
    b = _deterministic_sample(prompts, 5, seed=42)
    assert [p.id for p in a] == [p.id for p in b]


def test_sample_n_varies_with_seed() -> None:
    prompts = _build_prompts(20)
    a = _deterministic_sample(prompts, 5, seed=42)
    b = _deterministic_sample(prompts, 5, seed=43)
    assert [p.id for p in a] != [p.id for p in b]


def test_sample_n_preserves_suite_order() -> None:
    """The sampled subset is reordered to follow the original suite ordering."""
    prompts = _build_prompts(20)
    out = _deterministic_sample(prompts, 5, seed=42)
    indices = [int(p.id[1:]) for p in out]
    assert indices == sorted(indices)


def test_sample_n_warns_when_n_exceeds_suite_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Out-of-range N falls back to using every prompt and logs a warning."""
    prompts = _build_prompts(3)
    with caplog.at_level(logging.WARNING, logger="lre.runner"):
        out = _deterministic_sample(prompts, 5, seed=42)
    assert len(out) == 3
    assert any("exceeds" in r.getMessage() for r in caplog.records)


def test_sample_n_equal_to_suite_size_returns_all() -> None:
    prompts = _build_prompts(3)
    out = _deterministic_sample(prompts, 3, seed=42)
    assert len(out) == 3
    assert {p.id for p in out} == {p.id for p in prompts}


def test_sample_via_run_eval_yields_n_responses() -> None:
    prompts = _build_prompts(10)
    client = FakeModelClient(name="fake-1b", refusal_rate=0.5, seed=1)
    config = RunConfig(model="fake-1b", suites=["demo"], seed=42)
    responses = asyncio.run(run_eval(client, "demo", config, sample_n=3, prompts=prompts))
    assert len(responses) == 3


def test_cli_run_sample_writes_suite_marker(tmp_path: Path) -> None:
    """The CLI suffixes the suite name with ``[sampled N/M, seed=K]``."""
    out = tmp_path / "r.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--adapter",
            "fake",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--sample",
            "3",
            "--seed",
            "42",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["total"] == 3
    assert "sampled" in payload[0]["suite"]
    assert "seed=42" in payload[0]["suite"]


def test_cli_run_sample_is_byte_stable(tmp_path: Path) -> None:
    """Two ``lre run --sample`` invocations with same args must yield same totals."""
    runner = CliRunner()
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    args_a = [
        "run",
        "--adapter",
        "fake",
        "--model",
        "fake-1b",
        "--suite",
        "harmful_helpful",
        "--sample",
        "3",
        "--seed",
        "7",
        "--out",
        str(out_a),
    ]
    args_b = [*args_a[:-1], str(out_b)]
    a = runner.invoke(main, args_a)
    b = runner.invoke(main, args_b)
    assert a.exit_code == 0 and b.exit_code == 0
    pa = json.loads(out_a.read_text())
    pb = json.loads(out_b.read_text())
    assert pa[0]["refused"] == pb[0]["refused"]
    assert pa[0]["complied"] == pb[0]["complied"]
    assert pa[0]["suite"] == pb[0]["suite"]


def test_demo_with_sample_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--sample", "2", "--seed", "1"])
    assert result.exit_code == 0, result.output
