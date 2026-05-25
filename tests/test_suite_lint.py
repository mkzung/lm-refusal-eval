"""Tests for the suite JSONL linter."""

from __future__ import annotations

from pathlib import Path

from lre.suite_lint import lint_suite_file, lint_suite_lines


def test_linting_a_clean_suite_returns_no_issues() -> None:
    """The bundled ``harmful_helpful`` suite must lint clean."""
    import importlib.resources

    suite = importlib.resources.files("lre").joinpath("data", "prompts", "harmful_helpful.jsonl")
    issues = lint_suite_file(Path(str(suite)))
    assert issues == []


def test_linting_invalid_json_reports_error() -> None:
    issues = lint_suite_lines(["not valid json\n"])
    assert any("invalid JSON" in i.message for i in issues)


def test_linting_duplicate_id_is_caught() -> None:
    issues = lint_suite_lines(
        [
            '{"id": "p1", "text": "a", "category": "harmful"}\n',
            '{"id": "p1", "text": "b", "category": "harmful"}\n',
        ]
    )
    msgs = [i.message for i in issues]
    assert any("duplicate id" in m for m in msgs)


def test_linting_duplicate_text_is_caught() -> None:
    issues = lint_suite_lines(
        [
            '{"id": "p1", "text": "same body", "category": "harmful"}\n',
            '{"id": "p2", "text": "same body", "category": "harmful"}\n',
        ]
    )
    assert any("duplicate prompt text" in i.message for i in issues)


def test_linting_unknown_category_is_caught() -> None:
    issues = lint_suite_lines(['{"id": "p1", "text": "a", "category": "spicy"}\n'])
    assert any("unknown category" in i.message for i in issues)


def test_linting_missing_required_field_is_caught() -> None:
    issues = lint_suite_lines(
        ['{"id": "p1", "category": "harmful"}\n']  # missing text
    )
    assert any("schema error" in i.message for i in issues)


def test_linting_skips_blank_and_comment_lines() -> None:
    issues = lint_suite_lines(
        [
            "\n",
            "# header comment\n",
            '{"id": "p1", "text": "a", "category": "harmful"}\n',
        ]
    )
    assert issues == []


def test_linting_empty_suite_reports_error() -> None:
    issues = lint_suite_lines(["# only a comment\n"])
    assert any("no prompts" in i.message for i in issues)


def test_linting_rejects_non_object_row() -> None:
    issues = lint_suite_lines(["[1, 2, 3]\n"])
    assert any("not a JSON object" in i.message for i in issues)
