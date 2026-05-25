"""Run-provenance metadata for reproducibility.

Every :class:`~lre.state.EvalResult` produced by ``lre run`` carries a
``Provenance`` snapshot describing *how, where, and when* the result was
generated. A result JSON file from 2026 must be reproducible in 2027 —
without the snapshot, the only thing the consumer sees is the headline
payload, with no way to tell which version of the harness wrote it.

Design notes
------------

* ``schema_version`` is independent from ``lre_version``: the harness
  may iterate freely on internals while keeping the *output JSON*
  schema stable. External tooling should branch on
  ``schema_version`` first.
* ``hostname_hash`` is a SHA-256 prefix of the raw hostname. We never
  embed the raw hostname so result files can be shared safely; the
  fingerprint is still stable enough to cluster runs from the same
  machine.
* ``git_sha`` and ``git_dirty`` are populated by best-effort
  ``git rev-parse`` / ``git status --porcelain`` invocations. Outside
  a git work tree the fields are ``None`` / ``False`` rather than
  raising — the harness should still produce a result file when run
  from a plain pip install.
* The ``run_timestamp_utc`` field is the only piece of data here that
  changes per-run. Tests that compare two result-file payloads
  byte-for-byte must either strip provenance or pin the timestamp
  through a test seam (see :func:`collect_provenance`'s ``now`` arg).

v0.8 schema bump (``schema_version`` ⇒ ``"0.8"``)
-------------------------------------------------
Added fields:

* ``model_id`` — concrete client name (e.g. ``"Qwen2-0.5B-Instruct@chat"``)
  so a v0.8 result file is reproducible without re-reading the surrounding
  CLI invocation.
* ``temperature`` / ``max_tokens`` — sampling knobs forwarded from
  :class:`~lre.state.RunConfig`.
* ``suite_hash`` — SHA-256 of the canonical-JSONL bytes for the suite
  the result was produced over. Lets external tooling detect a silent
  suite edit between runs.
* ``judge_prompt_hash`` — SHA-256 of the LLM-judge prompt template.
* ``transformers_version`` — best-effort ``importlib.metadata`` lookup
  for the ``transformers`` package; ``None`` if not installed.
* ``torch_use_deterministic_algorithms`` — best-effort probe of
  ``torch.are_deterministic_algorithms_enabled()``; ``None`` if torch
  is not importable.

All fields default to ``None`` so v0.7 result files (which lack them)
keep loading via the optional schema.

v1.0 schema bump (``schema_version`` ⇒ ``"1.0"``)
-------------------------------------------------
The schema is now stable. ``lre reproduce`` requires every field
captured here to rebuild the original ``lre run`` invocation
byte-for-byte. Added fields:

* ``adapter`` — ``"fake"``, ``"hf"``, ``"openai"``, or ``"anthropic"``.
  Drives the ``lre reproduce --exec`` branch; without it the reproduce
  path cannot rebuild the right client.
* ``fake_refusal_rate`` — only set when ``adapter == "fake"``. The
  pre-v1.0 reproduce path hardcoded ``0.5``, so a run launched with
  ``--fake-refusal-rate 0.7`` was silently re-run at the wrong rate.
* ``sample_n`` — set when ``--sample N`` was passed. The pre-v1.0
  reproduce path stripped the ``[sampled N/M, seed=K]`` suite suffix
  and ran the FULL suite, silently destroying the sampled identity.
* ``judge_kind`` — ``"rule"`` or ``"llm"``. Groups in ``lre reproduce``
  now include the judge so two runs differing only in judge no longer
  collapse into a single reconstructed command.
* ``use_chat_template`` — only set when ``adapter == "hf"``. Operators
  iterating on base-vs-instruct checkpoints rely on this flag.
* ``api_key_env`` — only set when ``adapter`` is ``openai`` /
  ``anthropic``. Captures the operator's env-var choice so reproduce
  prints the exact ``--api-key-env`` value.
* ``base_url`` — only set when ``adapter == "openai"`` with a
  ``--base-url`` override (Azure / vLLM / OpenRouter / local OpenAI-
  compatible endpoints).
* ``max_concurrent`` — captured so reproduce parity holds for any
  reader inspecting concurrency-sensitive metrics.

All new fields default to ``None``. v0.7 / v0.8 / v0.9 result files
keep loading unchanged — the only place the bump bites is
``lre reproduce``, which now refuses pre-v1.0 results files with a
clean usage error pointing at the missing ``adapter`` field.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# v0.11 — typed enumerations for fields previously declared as ``str``.
# Pydantic validates the value at construction time; downstream code
# (notably :func:`lre.cli.reproduce`) was already branching on these
# strings, so locking them to a Literal prevents the "Unknown adapter"
# error path from being reachable via a malformed provenance JSON.
AdapterLiteral = Literal["fake", "hf", "openai", "anthropic"]
JudgeKindLiteral = Literal["rule", "llm"]


class Provenance(BaseModel):
    """Snapshot of how / where / when a result was produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # External tooling dispatches on this. Bumped to "1.0" in v0.10:
    # the v0.10 release adds the full set of CLI-input fields needed
    # for ``lre reproduce`` to rebuild the original invocation
    # byte-for-byte (adapter, fake_refusal_rate, sample_n, judge_kind,
    # use_chat_template, api_key_env, base_url, max_concurrent). The
    # schema is now stable enough to carry a non-pre-release version.
    schema_version: str = "1.0"
    lre_version: str
    python_version: str
    platform: str
    hostname_hash: str
    git_sha: str | None = None
    git_dirty: bool = False
    run_timestamp_utc: str
    seed: int
    # v0.8 additions — all optional so v0.7 JSON loads unchanged.
    model_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    suite_hash: str | None = None
    judge_prompt_hash: str | None = None
    transformers_version: str | None = None
    torch_use_deterministic_algorithms: bool | None = None
    # v1.0 additions — full CLI-input capture for ``lre reproduce``.
    # All optional so v0.7 / v0.8 / v0.9 JSON loads unchanged; the
    # reproduce path is the only one that requires them and surfaces
    # a clean error when ``adapter`` is absent.
    adapter: AdapterLiteral | None = None
    fake_refusal_rate: float | None = None
    sample_n: int | None = None
    judge_kind: JudgeKindLiteral | None = None
    use_chat_template: bool | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    max_concurrent: int | None = None


def _safe_lre_version() -> str:
    """Return the installed ``lm-refusal-eval`` version, falling back gracefully.

    ``importlib.metadata.version`` raises ``PackageNotFoundError`` when
    the package is not installed (e.g. running directly from a source
    checkout without a wheel/editable install). In that case we read the
    in-source ``__version__`` to keep provenance accurate during dev.
    """
    try:
        return metadata.version("lm-refusal-eval")
    except metadata.PackageNotFoundError:  # pragma: no cover - dev fallback
        from lre import __version__ as src_version

        return src_version


def _short_python_version() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def _hash_hostname(hostname: str) -> str:
    """SHA-256(hostname) truncated to 16 hex chars — privacy-respecting fingerprint."""
    return hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:16]


def _git_sha(cwd: Path | None = None) -> str | None:
    """Return the current commit SHA, or ``None`` if outside a git work tree.

    Uses ``git rev-parse HEAD`` so this is fast even on large repos. We
    deliberately do not catch ``FileNotFoundError`` from a missing ``git``
    binary specifically — the broad ``OSError`` clause covers it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _git_dirty(cwd: Path | None = None) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    return bool(out.stdout.strip())


def _utc_now_iso(now: _dt.datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp like ``2026-05-23T17:00:00Z``.

    v0.9 honors the ``SOURCE_DATE_EPOCH`` environment variable (the
    reproducible-builds convention): when set to a Unix epoch integer
    AND no explicit ``now`` test seam is passed, the timestamp is
    derived from that epoch rather than wall-clock time. This makes
    ``lre run`` byte-identical across reruns when the operator pins
    the epoch.
    """
    if now is None:
        epoch_env = os.environ.get("SOURCE_DATE_EPOCH")
        if epoch_env is not None:
            try:
                epoch = int(epoch_env)
            except ValueError:
                logger.warning("Ignoring SOURCE_DATE_EPOCH=%r (not an integer)", epoch_env)
            else:
                # Negative epochs would yield pre-1970 timestamps that
                # crash on Windows ``fromtimestamp`` and are never what
                # the operator intended. Reject and fall through to
                # wall-clock with a warning rather than failing the run.
                if epoch < 0:
                    logger.warning("Ignoring SOURCE_DATE_EPOCH=%r (negative epoch)", epoch_env)
                else:
                    when = _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc)
                    return when.strftime("%Y-%m-%dT%H:%M:%SZ")
    when = now if now is not None else _dt.datetime.now(_dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    # Truncate to seconds and use the ``Z`` military timezone suffix so
    # the rendered string is short and stable.
    return when.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_transformers_version() -> str | None:
    """Return the installed ``transformers`` version, or ``None`` if absent.

    Best-effort: any failure (package missing, broken metadata) returns
    ``None`` so the provenance probe never blocks a run.
    """
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("transformers version probe failed: %s", exc)
        return None


def _safe_torch_deterministic_flag() -> bool | None:
    """Probe :func:`torch.are_deterministic_algorithms_enabled`, swallowing errors.

    ``None`` means "torch is not importable" — the provenance consumer
    should treat that as "no torch in this run" rather than "torch
    present but non-deterministic". Best-effort.
    """
    try:
        import torch

        return bool(torch.are_deterministic_algorithms_enabled())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("torch deterministic-flag probe failed: %s", exc)
        return None


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data`` (full 64-char hex string).

    Exposed so :class:`~lre.runner.aggregate_results` and the
    :func:`collect_provenance` helper can share the same hashing
    convention without re-importing :mod:`hashlib` everywhere.
    """
    return hashlib.sha256(data).hexdigest()


def collect_provenance(
    seed: int,
    *,
    cwd: Path | None = None,
    now: _dt.datetime | None = None,
    hostname: str | None = None,
    git_sha_fn: Callable[[Path | None], str | None] | None = None,
    git_dirty_fn: Callable[[Path | None], bool] | None = None,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    suite_hash: str | None = None,
    judge_prompt_hash: str | None = None,
    adapter: AdapterLiteral | None = None,
    fake_refusal_rate: float | None = None,
    sample_n: int | None = None,
    judge_kind: JudgeKindLiteral | None = None,
    use_chat_template: bool | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    max_concurrent: int | None = None,
) -> Provenance:
    """Build a :class:`Provenance` snapshot for the current process.

    Parameters
    ----------
    seed:
        Seed of the run the snapshot describes — captured verbatim so
        the snapshot is self-contained.
    cwd:
        Optional working directory to probe for git metadata. Defaults
        to the current process working directory.
    now:
        Test seam — pass a fixed timestamp to make the snapshot stable
        across reruns. Defaults to ``datetime.now(UTC)``.
    hostname:
        Test seam — pass a fixed hostname instead of reading
        :func:`socket.gethostname`.
    git_sha_fn / git_dirty_fn:
        Test seams for the git probes. Default to the real subprocess
        implementations.
    model_id, temperature, max_tokens:
        v0.8 fields forwarded from the calling :class:`~lre.state.RunConfig`.
        Default to ``None`` so legacy callers that don't supply them keep
        producing valid snapshots.
    suite_hash:
        v0.8 — SHA-256 of the canonical suite bytes the result was
        produced over. Caller computes this (the provenance helper has
        no way to read the suite file itself without coupling).
    judge_prompt_hash:
        v0.8 — SHA-256 of the LLM-judge prompt template. ``None`` when
        the run used the rule judge.
    adapter, fake_refusal_rate, sample_n, judge_kind, use_chat_template,
    api_key_env, base_url, max_concurrent:
        v1.0 — full CLI-input capture so :func:`lre.cli.reproduce` can
        rebuild the original invocation byte-for-byte. All default to
        ``None`` so legacy callers stay valid; the reproduce path
        surfaces a clean error when ``adapter`` is absent.
    """
    raw_host = hostname if hostname is not None else socket.gethostname()
    sha_fn = git_sha_fn or _git_sha
    dirty_fn = git_dirty_fn or _git_dirty
    return Provenance(
        lre_version=_safe_lre_version(),
        python_version=_short_python_version(),
        platform=platform.platform(),
        hostname_hash=_hash_hostname(raw_host),
        git_sha=sha_fn(cwd),
        git_dirty=dirty_fn(cwd),
        run_timestamp_utc=_utc_now_iso(now),
        seed=seed,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        suite_hash=suite_hash,
        judge_prompt_hash=judge_prompt_hash,
        transformers_version=_safe_transformers_version(),
        torch_use_deterministic_algorithms=_safe_torch_deterministic_flag(),
        adapter=adapter,
        fake_refusal_rate=fake_refusal_rate,
        sample_n=sample_n,
        judge_kind=judge_kind,
        use_chat_template=use_chat_template,
        api_key_env=api_key_env,
        base_url=base_url,
        max_concurrent=max_concurrent,
    )
