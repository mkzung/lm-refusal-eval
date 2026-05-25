"""Tests for the JSONL prompt loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from lre.prompts import (
    InvalidSuiteName,
    SuiteNotFoundError,
    SuiteParseError,
    list_suites,
    load_suite,
)


def test_bundled_suites_are_listed() -> None:
    suites = list_suites()
    assert "harmful_helpful" in suites
    assert "jailbreak_styles" in suites
    assert "refusal_calibration" in suites


def test_load_suite_returns_expected_counts() -> None:
    assert len(load_suite("harmful_helpful")) == 15
    assert len(load_suite("jailbreak_styles")) == 10
    assert len(load_suite("refusal_calibration")) == 15


def test_load_suite_stamps_suite_name() -> None:
    prompts = load_suite("harmful_helpful")
    assert all(p.suite == "harmful_helpful" for p in prompts)


def test_load_unknown_suite_raises() -> None:
    with pytest.raises(SuiteNotFoundError):
        load_suite("does_not_exist")


def test_malformed_jsonl_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "broken.jsonl"
    bad.write_text("not a json line\n", encoding="utf-8")

    def fake_repo_dir() -> Path:
        return tmp_path

    monkeypatch.setattr("lre.prompts._repo_data_dir", fake_repo_dir)
    monkeypatch.setattr("lre.prompts._packaged_data_dir", lambda: None)

    with pytest.raises(SuiteParseError):
        load_suite("broken")


def test_schema_violation_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad_schema.jsonl"
    # Missing required field "text"
    bad.write_text('{"id": "p1", "category": "helpful"}\n', encoding="utf-8")

    monkeypatch.setattr("lre.prompts._repo_data_dir", lambda: tmp_path)
    monkeypatch.setattr("lre.prompts._packaged_data_dir", lambda: None)

    with pytest.raises(SuiteParseError):
        load_suite("bad_schema")


@pytest.mark.parametrize(
    "bad_name",
    [
        "../etc/passwd",
        "..",
        "foo/../bar",
        "/abs/path",
        "foo/bar",
        "foo\\bar",
        "with\x00null",
        "",
    ],
)
def test_load_suite_rejects_path_traversal_segments(bad_name: str) -> None:
    """the current implementation: ``load_suite`` refuses path-traversal / separator segments.

    Without this defence, ``lre run --suite ../etc/passwd`` would
    escape the bundled data dir on the first ``base / f"{suite}.jsonl"``
    join.
    """
    with pytest.raises(InvalidSuiteName):
        load_suite(bad_name)
