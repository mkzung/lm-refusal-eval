"""Tests for the Markdown / JSON reporters."""

from __future__ import annotations

from pathlib import Path

import pytest

from lre.report import (
    _escape_md,  # type: ignore[attr-defined]
    _fmt_ci,  # type: ignore[attr-defined]
    _fmt_rate,  # type: ignore[attr-defined]
    from_json,
    scaling_table,
    to_json,
    to_markdown,
)
from lre.state import EvalResult


def _sample_results() -> list[EvalResult]:
    return [
        EvalResult(
            model="qwen-mock-1.5b",
            suite="harmful_helpful",
            total=15,
            refused=10,
            complied=5,
            ambiguous=0,
            refusal_rate=10 / 15,
            latency_p50_s=0.1,
            latency_p99_s=0.2,
        ),
        EvalResult(
            model="qwen-mock-0.5b",
            suite="harmful_helpful",
            total=15,
            refused=5,
            complied=10,
            ambiguous=0,
            refusal_rate=5 / 15,
            latency_p50_s=0.05,
            latency_p99_s=0.09,
        ),
    ]


def test_json_round_trip_preserves_results() -> None:
    original = _sample_results()
    parsed = from_json(to_json(original))
    assert sorted(parsed, key=lambda r: r.model) == sorted(original, key=lambda r: r.model)


def test_json_is_byte_stable() -> None:
    blob_a = to_json(_sample_results())
    blob_b = to_json(_sample_results())
    assert blob_a == blob_b
    assert blob_a.endswith("\n")


def test_markdown_contains_expected_headers() -> None:
    md = to_markdown(_sample_results(), title="My report")
    assert "# My report" in md
    assert "| Model |" in md
    assert "| Suite |" in md
    assert "Refusal rate" in md
    # Rows sorted by (model, suite) — 0.5b row precedes 1.5b row.
    pos_05 = md.index("qwen-mock-0.5b")
    pos_15 = md.index("qwen-mock-1.5b")
    assert pos_05 < pos_15


def test_markdown_empty_input() -> None:
    md = to_markdown([], title="empty")
    assert "_No results._" in md


def test_scaling_table_pivots_by_size() -> None:
    rows = _sample_results()
    rows.append(
        EvalResult(
            model="qwen-mock-7b",
            suite="harmful_helpful",
            total=15,
            refused=12,
            complied=3,
            ambiguous=0,
            refusal_rate=12 / 15,
            latency_p50_s=0.1,
            latency_p99_s=0.2,
        )
    )
    table = scaling_table(rows)
    # 0.5 before 1.5 before 7 in the rendered text
    pos_05 = table.index("qwen-mock-0.5b")
    pos_15 = table.index("qwen-mock-1.5b")
    pos_7 = table.index("qwen-mock-7b")
    assert pos_05 < pos_15 < pos_7
    # Header includes the suite name
    assert "harmful_helpful" in table


def test_cached_fixture_round_trips() -> None:
    fixture = Path(__file__).parent / "fixtures" / "cached_run.json"
    blob = fixture.read_text(encoding="utf-8")
    results = from_json(blob)
    # The fixture covers three models x three suites = 9 rows.
    assert len(results) == 9
    assert {r.model for r in results} == {
        "qwen-mock-0.5b",
        "qwen-mock-1.5b",
        "qwen-mock-7b",
    }
    # Re-serializing should be byte-identical.
    assert to_json(results) == blob


def test_from_json_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        from_json('{"not": "an array"}')


# ---------------------------------------------------------------------------
# F-R2-T-16: per-category sub-table tests
# ---------------------------------------------------------------------------


def test_fmt_ci_renders_bracketed_pair() -> None:
    assert _fmt_ci(0.1, 0.2) == "[0.100, 0.200]"


def test_fmt_ci_renders_emdash_when_missing() -> None:
    assert _fmt_ci(None, 0.2) == "—"
    assert _fmt_ci(0.1, None) == "—"
    assert _fmt_ci(None, None) == "—"


def test_fmt_rate_renders_emdash_when_none() -> None:
    assert _fmt_rate(None) == "—"
    assert _fmt_rate(0.5) == "0.500"


def test_markdown_per_category_subtable_omitted_when_no_categories() -> None:
    """A report without any per-category data must NOT emit the
    sub-table heading.
    """
    rows = [
        EvalResult(
            model="m",
            suite="s",
            total=10,
            refused=5,
            complied=5,
            ambiguous=0,
            refusal_rate=0.5,
            latency_p50_s=0.0,
            latency_p99_s=0.0,
        )
    ]
    md = to_markdown(rows)
    assert "Refusal rate by prompt category" not in md


def test_markdown_per_category_subtable_emitted_with_alphabetical_columns() -> None:
    """When category data is present, the sub-table appears with
    categories sorted alphabetically.
    """
    rows = [
        EvalResult(
            model="m",
            suite="s",
            total=4,
            refused=2,
            complied=2,
            ambiguous=0,
            refusal_rate=0.5,
            refusal_rate_by_category={
                "helpful": 0.25,
                "harmful": 0.75,
                "borderline": None,
            },
            latency_p50_s=0.0,
            latency_p99_s=0.0,
        )
    ]
    md = to_markdown(rows)
    assert "Refusal rate by prompt category" in md
    # Categories appear in alphabetical order: borderline, harmful, helpful.
    pos_b = md.index("borderline")
    pos_h = md.index("harmful")
    pos_p = md.index("helpful")
    assert pos_b < pos_h < pos_p
    # The borderline column should render as the em-dash for the None value.
    assert "—" in md


def test_to_markdown_escapes_pipe_in_model_name() -> None:
    """F-R2-P3-21: a model name containing a pipe must be escaped so
    the table layout is not corrupted.
    """
    rows = [
        EvalResult(
            model="qwen-1.5b | finetune",
            suite="s",
            total=10,
            refused=5,
            complied=5,
            ambiguous=0,
            refusal_rate=0.5,
            latency_p50_s=0.0,
            latency_p99_s=0.0,
        )
    ]
    md = to_markdown(rows)
    assert r"qwen-1.5b \| finetune" in md
    # And the raw, unescaped form must not leak through.
    assert "qwen-1.5b | finetune |" not in md  # would corrupt the row


def test_to_markdown_strips_newlines_in_cells() -> None:
    rows = [
        EvalResult(
            model="multi\nline",
            suite="s",
            total=2,
            refused=1,
            complied=1,
            ambiguous=0,
            refusal_rate=0.5,
            latency_p50_s=0.0,
            latency_p99_s=0.0,
        )
    ]
    md = to_markdown(rows)
    # Cell should be flattened — no raw `\n` inside the model name.
    assert "multi line" in md
    # The line for that row must still be a single line.
    assert "multi\nline" not in md


def test_escape_md_idempotent_on_normal_input() -> None:
    """Strings without pipes or newlines pass through unchanged."""
    assert _escape_md("qwen-7b") == "qwen-7b"
    assert _escape_md("harmful_helpful") == "harmful_helpful"


def test_scaling_table_message_when_all_none() -> None:
    """F-R2-P2-14: when every row's refusal_rate is None, the table
    must use the explicit 'every model errored on every prompt' message
    rather than the generic '_No results._'.
    """
    rows = [
        EvalResult(
            model="m",
            suite="s",
            total=3,
            refused=0,
            complied=0,
            ambiguous=3,
            refusal_rate=None,
            latency_p50_s=0.0,
            latency_p99_s=0.0,
        )
    ]
    table = scaling_table(rows)
    assert "every model errored on every prompt" in table
