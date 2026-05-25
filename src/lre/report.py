"""Markdown + JSON reporters with byte-stable output.

We pin a few conventions so reruns produce identical bytes:

* JSON: ``sort_keys=True``, ``indent=2``, ``ensure_ascii=False``,
  ``allow_nan=False``, no trailing whitespace, single trailing newline.
  ``None`` refusal rates (every prompt errored) serialize as ``null``.
* Markdown: deterministic content + sort order; column widths float with
  the content (rendered via ``" | ".join``) but are reproducible across
  runs because the row ordering is sorted ``(model, suite)`` and the
  cell formatting is fixed. Missing values render as the em-dash sentinel
  ``"—"`` so tables stay aligned visually even when a cell is empty.
* :func:`scaling_table` parses an integer model size from the model name
  via the regex ``r"(\\d+(?:\\.\\d+)?)\\s*[bB]"``, which matches the
  ``0.5b`` / ``1.5b`` / ``7b`` / ``13b`` family of identifiers. Rows with
  ``refusal_rate is None`` are skipped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from lre.state import EvalResult

_MISSING = "—"

# ---------------------------------------------------------------------------
# Markdown escaping
# ---------------------------------------------------------------------------


def _escape_md(value: str) -> str:
    """Escape characters that would break the Markdown table layout.

    Specifically: ``|`` is escaped to ``\\|`` (cell separator) and any
    newline is collapsed to a single space (Markdown row terminator).
    Trailing whitespace is trimmed so the rendered cell stays compact.

    Cell text in the harness comes from model and suite names — neither
    should contain pipes or newlines in practice, but a user-supplied
    model identifier like ``"qwen-1.5b | finetune"`` would silently
    corrupt the table without this escape.
    """
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\r\n", " ").replace("\n", " ").replace("|", r"\|").rstrip()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(results: Sequence[EvalResult]) -> str:
    """Serialize results to a sorted, NaN-safe JSON string.

    Output format
    -------------
    The top-level value is always a JSON array of :class:`EvalResult`
    dicts. Each row carries its own ``provenance.schema_version`` when
    provenance is attached (``lre run``); rows produced by ``lre demo``
    omit the provenance footer entirely so the demo path stays
    byte-stable.

    The historical contract — a top-level JSON array — is preserved so
    existing scripts that ``jq '.[]|.refusal_rate'`` keep working. The
    ``schema_version`` discriminator that external tooling needs lives
    inside each row's provenance block.
    """
    payload = [r.model_dump(mode="json") for r in results]
    payload.sort(key=lambda row: (str(row["model"]), str(row["suite"])))
    rendered = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return rendered + "\n"


def from_json(blob: str) -> list[EvalResult]:
    """Inverse of :func:`to_json` — useful for the ``lre report`` CLI."""
    rows = json.loads(blob)
    if not isinstance(rows, list):
        msg = "expected a top-level JSON array of EvalResult dicts"
        raise TypeError(msg)
    return [EvalResult(**row) for row in rows]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


_HEADERS: tuple[str, ...] = (
    "Model",
    "Suite",
    "Total",
    "Refused",
    "Complied",
    "Ambig.",
    "Refusal rate",
    "95% CI",
    "p50 (s)",
    "p99 (s)",
)


def _fmt_rate(value: float | None) -> str:
    return _MISSING if value is None else f"{value:.3f}"


def _fmt_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return _MISSING
    return f"[{low:.3f}, {high:.3f}]"


def to_markdown(results: Sequence[EvalResult], title: str = "Refusal eval") -> str:
    """Render a deterministic Markdown report."""
    rows = sorted(results, key=lambda r: (r.model, r.suite))
    lines: list[str] = [f"# {title}", ""]
    if not rows:
        lines.append("_No results._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| " + " | ".join(_HEADERS) + " |")
    lines.append("|" + "|".join(["---"] * len(_HEADERS)) + "|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row.model),
                    _escape_md(row.suite),
                    str(row.total),
                    str(row.refused),
                    str(row.complied),
                    str(row.ambiguous),
                    _fmt_rate(row.refusal_rate),
                    _fmt_ci(row.refusal_rate_ci_low, row.refusal_rate_ci_high),
                    f"{row.latency_p50_s:.3f}",
                    f"{row.latency_p99_s:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Refusal rate = refused / (refused + complied). Errored generations "
        "are counted as 'Ambig.' and excluded from the denominator; runs "
        "where every prompt errored render as '—'."
    )
    lines.append("")
    # Per-category sub-table — only emitted when at least one row carries
    # category data, to keep the demo output compact.
    if any(row.refusal_rate_by_category for row in rows):
        lines.append("## Refusal rate by prompt category")
        lines.append("")
        categories: list[str] = sorted(
            {str(cat) for row in rows for cat in row.refusal_rate_by_category}
        )
        header = ["Model", "Suite", *[_escape_md(c) for c in categories]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            cells = [_escape_md(row.model), _escape_md(row.suite)]
            # Reflect the typed dict via its string-keyed view so we can
            # look up arbitrary category names without coercing types.
            cat_view: dict[str, float | None] = {
                str(k): v for k, v in row.refusal_rate_by_category.items()
            }
            for cat in categories:
                cells.append(_fmt_rate(cat_view.get(cat)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    footer = _provenance_footer(rows)
    if footer is not None:
        lines.append(footer)
        lines.append("")
    return "\n".join(lines)


def _provenance_footer(rows: Sequence[EvalResult]) -> str | None:
    """Render a one-line italic provenance footer, or ``None`` if absent.

    Uses the first row's provenance — every row produced by a single
    ``lre run`` invocation shares the same snapshot. ``lre demo``
    leaves provenance off entirely; those rows return ``None`` here
    so the byte-stable demo output is unaffected.
    """
    for row in rows:
        prov = row.provenance
        if prov is None:
            continue
        sha_part = "no-git" if prov.git_sha is None else f"git {prov.git_sha[:7]}"
        if prov.git_dirty:
            sha_part += "+dirty"
        # ``platform.platform()`` is verbose ("Linux-6.8.0-...-generic-x86_64-with-glibc2.39").
        # Use only the first dash-separated token to keep the footer compact.
        platform_short = prov.platform.split("-")[0].lower()
        # ``run_timestamp_utc`` is ``YYYY-MM-DDTHH:MM:SSZ`` — strip the
        # seconds so the footer stays one line.
        ts_short = prov.run_timestamp_utc[:16] + "Z"
        return (
            f"_Run: lre {prov.lre_version} · py {prov.python_version} · "
            f"{platform_short} · {sha_part} · seed={prov.seed} · {ts_short}_"
        )
    return None


# ---------------------------------------------------------------------------
# Scaling pivot
# ---------------------------------------------------------------------------


_SIZE_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]")


def _parse_size(model_name: str) -> float | None:
    """Extract the parameter-count size from a model identifier.

    Matches the ``N.NNb`` family of suffixes (``0.5b`` / ``1.5b`` / ``7b``
    / ``13b``). When multiple matches exist in the same string, the
    **rightmost** is preferred — HuggingFace repo names typically encode
    the canonical size as the trailing token (``my-org/qwen-7b`` is a 7B
    model; ``qwen-1.5b-finetune-7b`` would be a 7B model fine-tuned from a
    1.5B base, so the trailing ``7b`` is the authoritative size).
    """
    matches = list(_SIZE_REGEX.finditer(model_name))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1))
    except ValueError:  # pragma: no cover — regex guarantees numeric
        return None


def scaling_table(results: Sequence[EvalResult]) -> str:
    """Pivot refusal rate by (parsed-size, suite).

    Rows with ``refusal_rate is None`` are skipped — without a real rate
    they would force every column they touch into the missing-value
    sentinel and clutter the scaling pivot.

    Models without a parseable size (no ``N.NNb`` substring) are reported
    in an "Unknown size" section so they remain visible without polluting
    the sorted-by-size table.
    """
    sized: list[tuple[float, EvalResult]] = []
    unsized: list[EvalResult] = []
    for row in results:
        if row.refusal_rate is None:
            continue
        size = _parse_size(row.model)
        if size is None:
            unsized.append(row)
        else:
            sized.append((size, row))
    sized.sort(key=lambda pair: (pair[0], pair[1].model, pair[1].suite))

    suites = sorted({r.suite for r in results if r.refusal_rate is not None})
    lines: list[str] = ["# Refusal-rate scaling table", ""]
    if not sized and not unsized:
        # All inputs had refusal_rate=None — every prompt errored on every
        # row. The previous "_No results._" message was misleading because
        # the caller did supply rows; they just all failed.
        lines.append("_No rated results — every model errored on every prompt._")
        lines.append("")
        return "\n".join(lines)

    if sized:
        header = ["Size (B)", "Model", *[_escape_md(s) for s in suites]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        # Group by (size, model); within a model, populate columns per suite.
        per_model: dict[tuple[float, str], dict[str, EvalResult]] = {}
        for size, row in sized:
            per_model.setdefault((size, row.model), {})[row.suite] = row
        for (size, model), suite_map in sorted(per_model.items()):
            cells = [f"{size:g}", _escape_md(model)]
            for suite in suites:
                cell = suite_map.get(suite)
                cells.append(_fmt_rate(cell.refusal_rate) if cell is not None else _MISSING)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if unsized:
        lines.append("## Unknown size")
        lines.append("")
        header = ["Model", *[_escape_md(s) for s in suites]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        per_model_u: dict[str, dict[str, EvalResult]] = {}
        for row in unsized:
            per_model_u.setdefault(row.model, {})[row.suite] = row
        for model in sorted(per_model_u):
            cells = [_escape_md(model)]
            for suite in suites:
                cell = per_model_u[model].get(suite)
                cells.append(_fmt_rate(cell.refusal_rate) if cell is not None else _MISSING)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)
