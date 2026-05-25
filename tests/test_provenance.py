"""Tests for :mod:`lre.provenance`.

The provenance snapshot embedded in every ``EvalResult`` produced by
``lre run`` is the cornerstone of the v0.5 reproducibility story. These
tests exercise the collector's git probes, the privacy-preserving
hostname hash, and the optional test seams that make the snapshot
stable enough to assert in unit tests.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

from lre.provenance import (
    Provenance,
    _hash_hostname,
    _utc_now_iso,
    collect_provenance,
)
from lre.report import from_json, to_json
from lre.runner import aggregate_results
from lre.state import EvalResult, RawResponse, RefusalLabel


def test_collect_provenance_inside_git_repo(tmp_path: Path) -> None:
    """Inside an initialised repo, ``git_sha`` should match ``git rev-parse HEAD``."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    prov = collect_provenance(seed=1, cwd=tmp_path)
    assert prov.git_sha == sha
    assert prov.git_dirty is False


def test_collect_provenance_dirty_flag(tmp_path: Path) -> None:
    """Adding an untracked file should flip ``git_dirty`` to True."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "scratch.txt").write_text("dirty")
    prov = collect_provenance(seed=2, cwd=tmp_path)
    assert prov.git_dirty is True


def test_collect_provenance_outside_git_repo(tmp_path: Path) -> None:
    """A plain directory with no git metadata should yield ``git_sha is None``."""
    prov = collect_provenance(seed=3, cwd=tmp_path)
    assert prov.git_sha is None
    assert prov.git_dirty is False


def test_collect_provenance_hostname_is_hashed_not_raw() -> None:
    """The raw hostname must NEVER appear in the snapshot — only a SHA prefix."""
    fake_host = "secret-hostname-12345.example.internal"
    prov = collect_provenance(seed=4, hostname=fake_host)
    assert fake_host not in prov.hostname_hash
    assert fake_host not in prov.model_dump_json()
    expected_prefix = hashlib.sha256(fake_host.encode()).hexdigest()[:16]
    assert prov.hostname_hash == expected_prefix
    # Sanity check: 16 hex chars only.
    assert len(prov.hostname_hash) == 16
    assert all(c in "0123456789abcdef" for c in prov.hostname_hash)


def test_collect_provenance_seed_is_captured() -> None:
    prov = collect_provenance(seed=12345)
    assert prov.seed == 12345


def test_collect_provenance_pinned_timestamp_is_stable() -> None:
    """Two snapshots with the same ``now=`` seam should be identical."""
    fixed = _dt.datetime(2026, 5, 23, 17, 0, 0, tzinfo=_dt.timezone.utc)
    a = collect_provenance(seed=1, now=fixed, hostname="stable.host")
    b = collect_provenance(seed=1, now=fixed, hostname="stable.host")
    # git fields can vary depending on the test's cwd; ignore them and
    # focus on the deterministic parts of the snapshot.
    pure_a = a.model_copy(update={"git_sha": None, "git_dirty": False})
    pure_b = b.model_copy(update={"git_sha": None, "git_dirty": False})
    assert pure_a == pure_b
    assert a.run_timestamp_utc == "2026-05-23T17:00:00Z"


def test_utc_now_iso_naive_datetime_is_assumed_utc() -> None:
    naive = _dt.datetime(2026, 1, 1, 12, 0, 0)
    rendered = _utc_now_iso(naive)
    assert rendered == "2026-01-01T12:00:00Z"


def test_hash_hostname_is_deterministic() -> None:
    a = _hash_hostname("host-A")
    b = _hash_hostname("host-A")
    c = _hash_hostname("host-B")
    assert a == b
    assert a != c


def test_eval_result_json_roundtrip_with_provenance() -> None:
    """to_json -> from_json preserves the provenance footer exactly."""
    fixed = _dt.datetime(2026, 5, 23, 17, 0, 0, tzinfo=_dt.timezone.utc)
    prov = collect_provenance(seed=42, now=fixed, hostname="rt.example")
    responses = [
        RawResponse(
            prompt_id="p1",
            model="m",
            output="ok",
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        ),
    ]
    labels = [
        RefusalLabel(prompt_id="p1", model="m", refused=False, confidence=0.0, judge="rule"),
    ]
    result = aggregate_results(
        model="m",
        suite="s",
        responses=responses,
        labels=labels,
        provenance=prov,
    )
    assert result.provenance is not None
    assert result.provenance.seed == 42
    blob = to_json([result])
    parsed = from_json(blob)
    assert parsed[0].provenance == result.provenance


def test_legacy_v04_result_json_loads_without_provenance() -> None:
    """A v0.4 result file (no ``provenance`` field) must load cleanly."""
    legacy = json.dumps(
        [
            {
                "model": "old-model",
                "suite": "old-suite",
                "total": 1,
                "refused": 0,
                "complied": 1,
                "ambiguous": 0,
                "refusal_rate": 0.0,
                "refusal_rate_ci_low": 0.0,
                "refusal_rate_ci_high": 0.9,
                "refusal_rate_by_category": {},
                "latency_p50_s": 0.0,
                "latency_p99_s": 0.0,
                # NB: no provenance key at all.
            }
        ]
    )
    rows = from_json(legacy)
    assert len(rows) == 1
    assert rows[0].provenance is None
    assert isinstance(rows[0], EvalResult)


def test_schema_version_is_1_0_string() -> None:
    """External tooling dispatches on schema_version — pin the value.

    v0.10 bumps the schema to ``"1.0"`` alongside the full CLI-input
    capture (adapter, fake_refusal_rate, sample_n, judge_kind,
    use_chat_template, api_key_env, base_url, max_concurrent). The
    bump signals that the harness contract is now stable.
    """
    prov = collect_provenance(seed=1)
    assert prov.schema_version == "1.0"


def test_provenance_v08_new_fields_default_to_none() -> None:
    """The v0.8 fields default to ``None`` so legacy ``collect_provenance``
    calls (no ``model_id`` / ``suite_hash`` / etc.) keep working.
    """
    prov = collect_provenance(seed=1)
    assert prov.model_id is None
    assert prov.temperature is None
    assert prov.max_tokens is None
    assert prov.suite_hash is None
    assert prov.judge_prompt_hash is None


def test_provenance_v08_accepts_explicit_fields() -> None:
    """When provided, v0.8 fields round-trip into the snapshot."""
    prov = collect_provenance(
        seed=1,
        model_id="Qwen2-0.5B-Instruct@chat",
        temperature=0.7,
        max_tokens=256,
        suite_hash="a" * 64,
        judge_prompt_hash="b" * 64,
    )
    assert prov.model_id == "Qwen2-0.5B-Instruct@chat"
    assert prov.temperature == 0.7
    assert prov.max_tokens == 256
    assert prov.suite_hash == "a" * 64
    assert prov.judge_prompt_hash == "b" * 64


def test_provenance_v07_json_loads_unchanged() -> None:
    """A v0.7 provenance JSON (no new fields) must still parse.

    Backward compat: external tooling holding v0.7 result files must
    not break when upgrading the harness.
    """
    legacy = {
        "schema_version": "0.7",
        "lre_version": "0.7.0",
        "python_version": "3.11.0",
        "platform": "Linux",
        "hostname_hash": "abc1234567890def",
        "run_timestamp_utc": "2026-05-23T17:00:00Z",
        "seed": 42,
        "git_sha": None,
        "git_dirty": False,
    }
    prov = Provenance(**legacy)
    assert prov.schema_version == "0.7"
    assert prov.model_id is None


def test_aggregate_results_with_provenance_requires_seed_or_snapshot() -> None:
    """``with_provenance=True`` must reject calls that don't supply either input."""
    import pytest

    responses: list[RawResponse] = []
    labels: list[RefusalLabel] = []
    with pytest.raises(ValueError, match="seed"):
        aggregate_results(
            model="m",
            suite="s",
            responses=responses,
            labels=labels,
            with_provenance=True,
        )


def test_aggregate_results_with_provenance_collects_snapshot() -> None:
    """``with_provenance=True`` + ``seed=`` actually attaches a snapshot."""
    response = RawResponse(
        prompt_id="p1",
        model="m",
        output="ok",
        generation_seconds=0.01,
        timestamp=0,
        seed=7,
    )
    label = RefusalLabel(
        prompt_id="p1",
        model="m",
        refused=False,
        confidence=0.0,
        judge="rule",
    )
    result = aggregate_results(
        model="m",
        suite="s",
        responses=[response],
        labels=[label],
        with_provenance=True,
        seed=7,
    )
    assert result.provenance is not None
    assert result.provenance.seed == 7


def test_provenance_respects_source_date_epoch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """v0.9 (P0-5): SOURCE_DATE_EPOCH pins the provenance timestamp.

    The README claim of "byte-identical reruns" only held for ``lre demo``;
    ``lre run`` provenance changed every invocation because of the wall-clock
    timestamp. Honoring the reproducible-builds convention closes that gap:
    set ``SOURCE_DATE_EPOCH`` to a fixed Unix epoch and two
    ``collect_provenance`` calls produce identical ``run_timestamp_utc``.
    """
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    # 1700000000 == 2023-11-14T22:13:20Z.
    first = collect_provenance(
        seed=1, hostname="h", git_sha_fn=lambda _c: None, git_dirty_fn=lambda _c: False
    )
    second = collect_provenance(
        seed=1, hostname="h", git_sha_fn=lambda _c: None, git_dirty_fn=lambda _c: False
    )
    assert first.run_timestamp_utc == second.run_timestamp_utc
    assert first.run_timestamp_utc == "2023-11-14T22:13:20Z"


def test_provenance_ignores_malformed_source_date_epoch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A non-integer SOURCE_DATE_EPOCH falls back to wall-clock; never raises."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-a-number")
    # Should not raise; just emit a warning and use wall clock.
    prov = collect_provenance(
        seed=1, hostname="h", git_sha_fn=lambda _c: None, git_dirty_fn=lambda _c: False
    )
    assert prov.run_timestamp_utc  # non-empty ISO timestamp


def test_provenance_ignores_negative_source_date_epoch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """v0.10: a negative SOURCE_DATE_EPOCH falls back to wall-clock; never raises.

    Pre-1970 timestamps crash on Windows ``fromtimestamp`` and are never
    what the operator intended. The helper warns and falls through.
    """
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "-1")
    prov = collect_provenance(
        seed=1, hostname="h", git_sha_fn=lambda _c: None, git_dirty_fn=lambda _c: False
    )
    assert prov.run_timestamp_utc  # non-empty ISO timestamp


def test_provenance_v10_new_fields_default_to_none() -> None:
    """v1.0 fields default to ``None`` so legacy callers stay valid."""
    prov = collect_provenance(seed=1)
    assert prov.adapter is None
    assert prov.fake_refusal_rate is None
    assert prov.sample_n is None
    assert prov.judge_kind is None
    assert prov.use_chat_template is None
    assert prov.api_key_env is None
    assert prov.base_url is None
    assert prov.max_concurrent is None


def test_provenance_v10_accepts_full_cli_capture() -> None:
    """When provided, v1.0 fields round-trip into the snapshot."""
    prov = collect_provenance(
        seed=1,
        adapter="fake",
        fake_refusal_rate=0.7,
        sample_n=3,
        judge_kind="rule",
        use_chat_template=True,
        api_key_env="MY_KEY",
        base_url="https://api.example.com/v1",
        max_concurrent=8,
    )
    assert prov.adapter == "fake"
    assert prov.fake_refusal_rate == 0.7
    assert prov.sample_n == 3
    assert prov.judge_kind == "rule"
    assert prov.use_chat_template is True
    assert prov.api_key_env == "MY_KEY"
    assert prov.base_url == "https://api.example.com/v1"
    assert prov.max_concurrent == 8


def test_utc_now_iso_explicit_now_overrides_source_date_epoch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An explicit ``now=`` test seam wins over SOURCE_DATE_EPOCH.

    The test seam exists so unit tests can pin a specific moment; the
    env-var convention is for operator-facing reproducibility. Explicit
    arguments must beat environmental defaults.
    """
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    explicit = _dt.datetime(2030, 1, 1, 12, 0, 0, tzinfo=_dt.timezone.utc)
    assert _utc_now_iso(now=explicit) == "2030-01-01T12:00:00Z"


def test_provenance_is_frozen() -> None:
    """Provenance snapshots must be immutable — they document a moment in time."""
    import pytest
    from pydantic import ValidationError

    prov = Provenance(
        lre_version="0.5.0",
        python_version="3.11.0",
        platform="Linux",
        hostname_hash="abc1234567890def",
        run_timestamp_utc="2026-05-23T17:00:00Z",
        seed=1,
    )
    with pytest.raises(ValidationError):
        prov.seed = 2  # type: ignore[misc]
