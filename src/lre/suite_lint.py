"""Lightweight JSONL suite-file linter.

Why a separate module? :func:`lre.prompts.load_suite` raises on the *first*
schema error, which is the right behaviour for runtime but a poor UX for
suite authors who want a single pass over the file showing every issue.
The linter here collects all issues without short-circuiting.

Checked invariants
------------------
* Every non-empty / non-comment line is valid JSON.
* Every row parses cleanly into a :class:`Prompt`.
* ``id`` is unique within the file.
* ``text`` is unique within the file (no duplicated prompt bodies).
* ``category`` is one of the allowed literals (this is enforced by
  Pydantic, but the linter surfaces a friendlier message).

The linter is intentionally permissive about ``notes`` and the ``suite``
field — both are optional / auto-stamped at load time.
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from lre.state import Prompt, PromptCategory

Severity = Literal["error", "warning"]

# Pull the canonical category set straight from the
# :data:`lre.state.PromptCategory` Literal so the two declarations
# cannot drift. ``typing.get_args`` returns the tuple of literal values
# in declaration order; freeze into a ``frozenset`` for cheap ``in``.
_VALID_CATEGORIES: frozenset[str] = frozenset(typing.get_args(PromptCategory))


@dataclass(frozen=True)
class LintIssue:
    """A single issue raised by :func:`lint_suite_file`.

    ``line_no`` is ``None`` for file-level issues (duplicate-id pairs
    where the second occurrence carries the line number, etc.).
    """

    severity: Severity
    message: str
    line_no: int | None = None


def lint_suite_lines(lines: Iterable[str]) -> list[LintIssue]:
    """Run all checks on an iterable of raw JSONL lines.

    The iterable is consumed exactly once. Lines are 1-indexed in
    diagnostics so the output is consumable by ``$EDITOR +<line>``.
    """
    issues: list[LintIssue] = []
    seen_ids: dict[str, int] = {}
    # Track prompt-body duplicates by SHA-256-prefix hash, not by raw
    # text. A 10k-prompt suite with ~2 KB bodies would otherwise hold
    # ~20 MB of strings live; the 16-byte digest is ~3 orders of
    # magnitude smaller and still has negligible collision risk on the
    # scales the linter is asked to handle (the linter raises an issue
    # on collision, so a false positive surfaces in CI immediately).
    seen_texts: dict[bytes, int] = {}
    for line_no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            issues.append(
                LintIssue(severity="error", message=f"invalid JSON: {exc.msg}", line_no=line_no)
            )
            continue
        if not isinstance(payload, dict):
            issues.append(
                LintIssue(
                    severity="error",
                    message=f"row is not a JSON object (got {type(payload).__name__})",
                    line_no=line_no,
                )
            )
            continue
        cat = payload.get("category")
        if cat is not None and cat not in _VALID_CATEGORIES:
            issues.append(
                LintIssue(
                    severity="error",
                    message=(
                        f"unknown category {cat!r}; expected one of {sorted(_VALID_CATEGORIES)}"
                    ),
                    line_no=line_no,
                )
            )
            # Don't fall through to Prompt validation; the literal mismatch
            # would produce a redundant error.
            continue
        payload.setdefault("suite", "_lint")
        try:
            prompt = Prompt(**payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ())) or "<root>"
            issues.append(
                LintIssue(
                    severity="error",
                    message=f"schema error at {loc}: {first.get('msg')}",
                    line_no=line_no,
                )
            )
            continue
        if prompt.id in seen_ids:
            issues.append(
                LintIssue(
                    severity="error",
                    message=(
                        f"duplicate id {prompt.id!r} (first seen on line {seen_ids[prompt.id]})"
                    ),
                    line_no=line_no,
                )
            )
        else:
            seen_ids[prompt.id] = line_no
        text_digest = hashlib.sha256(prompt.text.encode("utf-8")).digest()[:16]
        if text_digest in seen_texts:
            issues.append(
                LintIssue(
                    severity="error",
                    message=(
                        f"duplicate prompt text (first seen on line {seen_texts[text_digest]})"
                    ),
                    line_no=line_no,
                )
            )
        else:
            seen_texts[text_digest] = line_no
    if not seen_ids:
        issues.append(
            LintIssue(
                severity="error",
                message="suite contains no prompts",
                line_no=None,
            )
        )
    return issues


def lint_suite_file(path: Path) -> list[LintIssue]:
    """Read ``path`` and run :func:`lint_suite_lines` on its contents.

    Uses ``encoding="utf-8-sig"`` so a UTF-8 byte-order mark at the start
    of the file is stripped instead of breaking the very first JSON parse.

    The file is consumed via line-by-line iteration so the whole body
    never materialises in memory — important for the 10k+ prompt
    suites typical of scaling-laws work.
    """
    with path.open("r", encoding="utf-8-sig") as handle:
        return lint_suite_lines(handle)
