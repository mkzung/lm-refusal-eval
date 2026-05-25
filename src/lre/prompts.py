"""Loader for bundled JSONL prompt suites.

Suites live inside the package at ``src/lre/data/prompts/<name>.jsonl``.
They are shipped *inside* the wheel so ``importlib.resources.files("lre")``
returns them after ``pip install``. The loader also accepts an alternate
``<repo>/data/prompts`` location to keep legacy editable installs working,
but the canonical location is in-package.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

from lre.state import Prompt


class SuiteNotFoundError(FileNotFoundError):
    """Raised when a named suite cannot be located on disk."""


class SuiteParseError(ValueError):
    """Raised when a JSONL line cannot be parsed into a :class:`Prompt`."""


class InvalidSuiteName(ValueError):
    """Raised when a suite name contains path-separator or traversal segments.

    A suite name is a bare identifier — ``"harmful_helpful"``, never
    ``"../etc/passwd"``. Rejecting traversal segments up front prevents
    ``lre run --suite ../../somewhere`` from reading arbitrary files
    when callers point the loader at a writable data directory.
    """


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------


def _repo_data_dir() -> Path:
    """Return ``<repo>/data/prompts`` for legacy compatibility (may not exist)."""
    return Path(__file__).resolve().parents[2] / "data" / "prompts"


def _packaged_data_dir() -> Path | None:
    """Return ``<site-packages>/lre/data/prompts`` if it exists.

    This is the canonical location: ``src/lre/data/prompts`` in the source
    tree, and ``site-packages/lre/data/prompts`` after a wheel install.
    """
    try:
        anchor = resources.files("lre").joinpath("data").joinpath("prompts")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    # importlib.resources can return MultiplexedPath; cast to Path if possible.
    try:
        candidate = Path(str(anchor))
    except TypeError:
        return None
    return candidate if candidate.is_dir() else None


def _candidate_dirs() -> Iterable[Path]:
    packaged = _packaged_data_dir()
    if packaged is not None:
        yield packaged
    legacy = _repo_data_dir()
    if legacy.is_dir():
        yield legacy


_FORBIDDEN_SUITE_SEGMENTS: tuple[str, ...] = ("/", "\\", "\x00", "..")


def _validate_suite_name(suite: str) -> None:
    """Reject suite names that contain path-separator or traversal segments.

    A suite name is a bare identifier — ``"harmful_helpful"``, never
    ``"../etc/passwd"``. Without this check, ``--suite ../foo`` would
    escape the bundled data dir on the first ``base / f"{suite}.jsonl"``
    join and read arbitrary on-disk JSONL files. ``pathlib.Path`` does
    not normalise the join on its own, so the policy is enforced here.
    """
    if not isinstance(suite, str) or not suite:
        msg = "suite name must be a non-empty string"
        raise InvalidSuiteName(msg)
    for segment in _FORBIDDEN_SUITE_SEGMENTS:
        if segment in suite:
            msg = (
                f"invalid suite name {suite!r}: must not contain "
                f"path separators, null bytes, or traversal segments"
            )
            raise InvalidSuiteName(msg)
    # Defence-in-depth: reject absolute paths even when the platform
    # would not split on the segments above (e.g. a Windows drive letter
    # like ``C:foo``).
    if Path(suite).is_absolute():
        msg = f"invalid suite name {suite!r}: absolute paths are not allowed"
        raise InvalidSuiteName(msg)


def _resolve(suite: str) -> Path:
    _validate_suite_name(suite)
    for base in _candidate_dirs():
        path = base / f"{suite}.jsonl"
        if not path.is_file():
            continue
        # Belt-and-braces: after the join, verify the resolved path is
        # still under the candidate dir. Cheap symlink and traversal
        # check on platforms where the segment-level validation above
        # somehow missed the attack surface.
        try:
            resolved = path.resolve()
            base_resolved = base.resolve()
        except OSError:
            continue
        if not _is_relative_to(resolved, base_resolved):
            msg = f"invalid suite name {suite!r}: resolved path escapes the bundled data directory"
            raise InvalidSuiteName(msg)
        return path
    available = ", ".join(sorted(list_suites())) or "(none)"
    msg = f"suite '{suite}' not found; available: {available}"
    raise SuiteNotFoundError(msg)


def _is_relative_to(child: Path, parent: Path) -> bool:
    """Backport of ``Path.is_relative_to`` for Python 3.10 compatibility.

    ``Path.is_relative_to`` exists on 3.9+ but matches the parent only
    via string-prefix; using ``relative_to`` + ``except ValueError`` is
    equivalent and works identically across our supported Python range.
    """
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_suites() -> list[str]:
    """Return every bundled suite name, sorted."""
    found: set[str] = set()
    for base in _candidate_dirs():
        if not base.is_dir():
            continue
        for path in base.glob("*.jsonl"):
            found.add(path.stem)
    return sorted(found)


def suite_bytes_hash(suite: str) -> str | None:
    """Return SHA-256 of the on-disk suite JSONL bytes, or ``None`` if missing.

    Used by :func:`lre.provenance.collect_provenance` (via the CLI) to
    embed a suite fingerprint in the v0.8 provenance footer so an
    external consumer can detect a silent suite edit between runs
    without re-reading the suite. Best-effort: file-not-found or
    read failures return ``None`` instead of raising — the run
    should not abort because we cannot hash a suite.
    """
    import hashlib

    try:
        path = _resolve(suite)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (SuiteNotFoundError, OSError):
        return None


def load_suite(suite: str) -> list[Prompt]:
    """Load and validate all prompts from a named JSONL suite.

    Raises
    ------
    SuiteNotFoundError
        If no file matches the suite name.
    SuiteParseError
        If any JSONL row is malformed or fails schema validation.
    """
    path = _resolve(suite)
    prompts: list[Prompt] = []
    # encoding="utf-8-sig" transparently strips a UTF-8 byte-order mark
    # if present so suites authored on Windows do not break JSON parsing.
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"{path.name}:{line_no}: invalid JSON: {exc.msg}"
                raise SuiteParseError(msg) from exc
            payload.setdefault("suite", suite)
            try:
                prompts.append(Prompt(**payload))
            except ValidationError as exc:
                msg = f"{path.name}:{line_no}: schema error: {exc.errors()[0]['msg']}"
                raise SuiteParseError(msg) from exc
    if not prompts:
        msg = f"suite '{suite}' contains no prompts"
        raise SuiteParseError(msg)
    return prompts
