"""Click CLI entry point.

Subcommands cover the eval lifecycle:

* ``lre demo``    — offline end-to-end run using the synthetic client.
* ``lre run``     — run a real model adapter against one or more suites.
* ``lre judge``   — rejudge a previously generated set of raw responses.
* ``lre report``  — render Markdown / JSON / scaling-table from cached results.
* ``lre compare`` — diff two result files, with proportion test + Wilson CI on Δ.
* ``lre lint``    — validate a suite JSONL file (categories, dup ids, etc.).
* ``lre kappa``   — Cohen's κ inter-judge agreement on two label files.
* ``lre did``     — paired (inner, outer) defense-in-depth refusal stats.

Subcommands return ``0`` on success and ``1`` on user-input error
(missing files, bad suite names, missing API keys, schema errors).
Programmatic failures (mypy bugs, real schema bugs in the harness) are
allowed to propagate — the CLI is meant for working researchers, not for
hiding genuine bugs.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import logging
import os
import sys
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import click
from pydantic import ValidationError

from lre import __version__
from lre.cache import SENTINEL_FILENAME, ResponseCache, is_lre_cache_dir
from lre.defense_in_depth import (
    PairedDefense,
    aggregate_paired_results,
    paired_label_responses,
    to_markdown_did,
)
from lre.judge import _JUDGE_PROMPT as _LLM_JUDGE_PROMPT_TEMPLATE
from lre.judge import LLMJudge, RuleBasedJudge
from lre.prompts import (
    InvalidSuiteName,
    SuiteNotFoundError,
    list_suites,
    load_suite,
    suite_bytes_hash,
)
from lre.provenance import AdapterLiteral, collect_provenance, hash_bytes
from lre.report import from_json, scaling_table, to_json, to_markdown
from lre.runner import aggregate_results, ajudge_responses, run_eval
from lre.state import EvalResult, RawResponse, RefusalLabel, RunConfig
from lre.stats import cohen_kappa, compute_proportion_diff_test, compute_wilson_ci
from lre.suite_lint import lint_suite_file
from lre.synthetic import FakeModelClient

if TYPE_CHECKING:  # pragma: no cover
    from lre.models.base import ModelClient


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_raw_responses(path: Path) -> list[dict[str, Any]]:
    """Load raw responses from either a JSON array or a JSONL stream.

    ``lre judge --in r.json`` historically accepted only a JSON array.
    ``lre run --dump-raw r.jsonl`` writes a JSONL stream so every
    response is recoverable line-by-line on partial reads. Both
    formats are valid input here.

    Raises :class:`click.UsageError` on read or parse failure.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read raw-response file {path}: {exc}"
        raise click.UsageError(msg) from exc
    text = text.strip()
    if not text:
        msg = f"Raw-response file {path} is empty"
        raise click.UsageError(msg)
    # JSON-array branch.
    if text[0] == "[":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = (
                f"Failed to parse raw-response file {path}: {exc.msg} "
                f"(line {exc.lineno}, col {exc.colno})"
            )
            raise click.UsageError(msg) from exc
        if not isinstance(payload, list):
            msg = f"Expected a JSON array in raw-response file {path}; got {type(payload).__name__}"
            raise click.UsageError(msg)
        return [row for row in payload if isinstance(row, dict)]
    # JSONL branch.
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Failed to parse raw-response file {path}: line {line_no}: {exc.msg}"
            raise click.UsageError(msg) from exc
        if not isinstance(obj, dict):
            msg = (
                f"Raw-response file {path}: line {line_no}: expected an "
                f"object, got {type(obj).__name__}"
            )
            raise click.UsageError(msg)
        rows.append(obj)
    return rows


def _load_json_list(path: Path, kind: str) -> list[Any]:
    """Parse ``path`` as a JSON array, surfacing user-friendly errors.

    Raises :class:`click.UsageError` (which the Click runtime renders as a
    clean one-line error, no traceback) on:

    * unreadable file / file-not-found
    * malformed JSON
    * a JSON value that is not a top-level array

    The ``kind`` parameter names what we expected (e.g. ``"results"``,
    ``"labels"``, ``"raw responses"``) so the error message points the
    user at the right argument.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read {kind} file {path}: {exc}"
        raise click.UsageError(msg) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse {kind} file {path}: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        raise click.UsageError(msg) from exc
    if not isinstance(payload, list):
        msg = f"Expected a JSON array in {kind} file {path}; got {type(payload).__name__}"
        raise click.UsageError(msg)
    return payload


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for the CLI.

    Default verbosity is WARNING. ``-v`` / ``--verbose`` opts into INFO
    (the previous default, which was noisy for ``lre demo`` and
    ``lre run`` smoke tests). DEBUG is reserved for future ``-vv``-style
    flags; we have not implemented that here because the default
    log surface already covers the operator's needs.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _is_stdout_sentinel(path: Path | str | None) -> bool:
    """Return ``True`` when ``path`` is the Unix-style ``-`` stdout sentinel.

    Following the standard CLI convention: ``--out -`` writes to stdout
    instead of a literal file named ``-``. Pre-v0.9 the harness created
    a file named ``-`` in the cwd, which was almost never what the
    operator intended.
    """
    if path is None:
        return False
    return str(path) == "-"


def _write_text_out(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` or stdout when ``path`` is ``-``."""
    if _is_stdout_sentinel(path):
        sys.stdout.write(content)
        sys.stdout.flush()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe_response_cache(cache_dir: Path, *, allow_symlinked: bool = False) -> ResponseCache:
    """Build a :class:`ResponseCache` with a clean CLI error on bad paths.

    Without this wrapper, ``--cache /nonexistent_parent/sub`` or
    ``--cache /dev/null`` crashes with a raw ``OSError`` /
    ``FileNotFoundError`` traceback. Wrap and re-raise as
    :class:`click.UsageError` so the operator gets a one-line message
    pointing at the bad argument.

    Leaf symlinks are refused by default — a stale or attacker-controlled
    symlink at the cache leaf would silently redirect cache writes
    elsewhere on disk. Pass ``allow_symlinked=True`` to opt in.

    v0.9 (R7): parent-path symlinks now WARN instead of REFUSE. v0.8
    compared ``absolute()`` against ``resolve()``, which fired in two
    common benign cases: (1) any path containing ``..`` segments
    (``absolute()`` preserves them, ``resolve()`` collapses them) and
    (2) any path on macOS that traverses ``/tmp`` (which is a system
    symlink to ``/private/tmp``). The original threat model was a
    user-controlled leaf symlink — that protection is preserved.
    Parent symlinks emit a warning so the operator can decide.
    """
    if not allow_symlinked:
        # Refuse a symlink at the LEAF — that is the user-controlled
        # attack surface. ``Path.is_symlink()`` is the right primitive
        # here; it inspects the leaf inode without following.
        if cache_dir.is_symlink():
            msg = (
                f"--cache {cache_dir} is a symlink; refusing to follow. "
                "Pass --allow-symlinked-cache to opt in."
            )
            raise click.UsageError(msg)

        # Parent-path symlinks: warn, do not refuse. ``os.path.normpath``
        # collapses ``..`` segments WITHOUT following symlinks, so the
        # comparison against ``resolve()`` isolates real symlinks in
        # the parent chain rather than ``..`` artifacts. We still warn
        # because /tmp → /private/tmp on macOS is benign, but an
        # unexpected redirect in a CI environment is worth flagging.
        try:
            absolute_normalized = Path(os.path.normpath(str(cache_dir.absolute())))
            resolved = cache_dir.resolve(strict=False)
        except OSError as exc:
            msg = f"Cannot resolve --cache {cache_dir}: {exc}"
            raise click.UsageError(msg) from exc
        if resolved != absolute_normalized:
            click.echo(
                f"WARNING: --cache {cache_dir} resolves to {resolved} via a "
                "symlink in a parent directory. Proceeding; pass "
                "--allow-symlinked-cache to silence this warning.",
                err=True,
            )
    try:
        return ResponseCache(cache_dir)
    except OSError as exc:
        msg = f"Cannot use --cache {cache_dir}: {exc}"
        raise click.UsageError(msg) from exc


def _confidence_callback(ctx: click.Context, param: click.Parameter, value: str) -> float:
    """Convert the ``--confidence`` choice string into a float.

    Restricting the CLI to ``0.90 / 0.95 / 0.99`` avoids the prior
    raw ``ValueError`` traceback produced by upstream stats helpers
    when handed an unsupported confidence level. Returning a ``float``
    keeps the downstream type signatures unchanged.
    """
    return float(value)


def _warn_nondeterministic_cache(temperature: float, allow: bool, cache_dir: Path | None) -> None:
    """Surface a warning when a cache is paired with a sampling temperature.

    At ``temperature > 0`` the model is stochastic — a cache write
    cements one realised sample and every subsequent hit replays it
    instead of redrawing. That is fine for byte-stable replay debugging
    but actively wrong for any analysis that relies on sampling
    variance. The flag ``--cache-allow-nondeterministic`` suppresses
    the warning so a researcher who knows what they are doing isn't
    forced through the prompt every run.
    """
    if cache_dir is None or temperature <= 0.0 or allow:
        return
    click.echo(
        f"WARNING: caching at temperature={temperature} cements one sample; "
        "cache hits replay it instead of resampling. Use temperature=0.0 for "
        "fully reproducible caching, or pass --cache-allow-nondeterministic "
        "to suppress this warning.",
        err=True,
    )


def _humanize_validation_error(exc: ValidationError) -> str:
    """Render a :class:`pydantic.ValidationError` into one short sentence per error.

    Pydantic's default formatting is a multi-line dump with type hints,
    URLs to the docs, and JSON Schema references — fine for library
    users, but noisy in a CLI surface where the operator usually wants
    "what did I type wrong?". This helper keeps just the field path and
    the human-readable message.
    """
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        msg = err.get("msg") or "invalid"
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


@click.group(help="Reproducible refusal-rate harness for open-weight LLMs.")
@click.version_option(version=__version__, prog_name="lre")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug-level logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _configure_logging(verbose)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


@main.command(help="Run the offline synthetic demo across all bundled suites.")
@click.option(
    "--refusal-rate",
    "--fake-refusal-rate",
    "refusal_rate",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.6,
    show_default=True,
    help=(
        "Target synthetic refusal rate for the fake client. "
        "(``--fake-refusal-rate`` is the canonical name used by ``lre run``; "
        "both work here.)"
    ),
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help="Deterministic seed mixed into every prompt hash.",
)
@click.option(
    "--model-name",
    default="fake-1b",
    show_default=True,
    help="Identifier reported in the output table.",
)
@click.option(
    "--sample",
    "sample_n",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Sample N (>=1) prompts per suite (deterministic in --seed). "
        "Provided for symmetry with ``lre run``."
    ),
)
def demo(
    refusal_rate: float,
    seed: int,
    model_name: str,
    sample_n: int | None,
) -> None:
    suites = list_suites()
    if not suites:
        click.echo("No bundled suites found. Reinstall the package.", err=True)
        sys.exit(1)
    client = FakeModelClient(name=model_name, refusal_rate=refusal_rate, seed=seed)
    results = asyncio.run(_demo_async(client, suites, seed, sample_n=sample_n))
    click.echo(to_markdown(results, title=f"lre demo — {model_name}"))
    # Print a next-steps hint after the table so a researcher
    # running ``lre demo`` for the first time knows what to do next. We
    # use click.secho so colour-capable terminals get a cyan header, but
    # fall back to plain text everywhere else. The hint is appended to
    # stdout (after the table) so byte-stability for the markdown table
    # itself is not affected — the entire stdout payload remains stable
    # across reruns because the hint contains no timestamps or RNG.
    click.secho("\nNext steps:", fg="cyan")
    click.echo(
        "  - lre run --adapter hf --model qwen-0.5b "
        "--model-id Qwen/Qwen2-0.5B-Instruct "
        "--suite harmful_helpful --out r.json"
    )
    click.echo(
        "  - lre run --adapter openai --model gpt-4o-mini --suite jailbreak_styles --out r.json"
    )
    click.echo("  - lre compare r1.json r2.json")
    click.echo("See README for the full guide.")


async def _demo_async(
    client: ModelClient,
    suites: list[str],
    seed: int,
    *,
    sample_n: int | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for suite in suites:
        config = RunConfig(model=client.name, suites=[suite], seed=seed)
        full_prompts = load_suite(suite)
        responses = await run_eval(
            client,
            suite,
            config,
            sample_n=sample_n,
            prompts=full_prompts,
        )
        if sample_n is not None and sample_n > 0:
            sampled_ids = [r.prompt_id for r in responses]
            full_by_id = {p.id: p for p in full_prompts}
            prompts = [full_by_id[pid] for pid in sampled_ids if pid in full_by_id]
        else:
            prompts = list(full_prompts)
        labels = await ajudge_responses(responses)
        suite_label = suite
        # v0.12: preserve the ``[sampled N/M, seed=K]`` suffix even when
        # ``--sample`` overshoots the suite size, adding a ``capped``
        # marker so downstream consumers can tell a sample-was-requested-
        # but-capped run apart from a true full-suite run. Pre-v0.12
        # dropped the suffix entirely on overshoot — exactly the case
        # where the user most needs to know their sample request was
        # truncated.
        if sample_n is not None and sample_n > 0:
            if len(prompts) < len(full_prompts):
                suite_label = f"{suite}[sampled {len(prompts)}/{len(full_prompts)}, seed={seed}]"
            elif sample_n > len(full_prompts):
                suite_label = (
                    f"{suite}[sampled {len(full_prompts)}/{len(full_prompts)}, seed={seed}, capped]"
                )
        results.append(
            aggregate_results(
                model=client.name,
                suite=suite_label,
                responses=responses,
                labels=labels,
                prompts=prompts,
                # Demo deliberately omits provenance so the demo path
                # stays byte-stable (the run_timestamp_utc would
                # otherwise drift each invocation, breaking the
                # `test_lre_demo_is_byte_stable_across_invocations`
                # invariant).
            )
        )
    return results


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _build_client(
    adapter: str,
    model: str,
    *,
    model_id: str | None,
    base_url: str | None,
    api_key_env: str | None,
    seed: int,
    fake_refusal_rate: float = 0.5,
    use_chat_template: bool = True,
) -> ModelClient:
    """Instantiate the requested adapter, failing fast on missing config."""
    if adapter == "fake":
        return FakeModelClient(name=model, seed=seed, refusal_rate=fake_refusal_rate)
    if adapter == "hf":
        from lre.models.hf_local import HFLocalClient

        if not model_id:
            msg = "--adapter hf requires --model-id <hf repo or local path>"
            raise click.UsageError(msg)
        return HFLocalClient(model_id=model_id, name=model, use_chat_template=use_chat_template)
    if adapter == "openai":
        from lre.models.openai_api import OpenAIClient

        env = api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env)
        if not api_key:
            msg = f"Set {env} before running --adapter openai"
            raise click.UsageError(msg)
        if base_url:
            return OpenAIClient(model=model, api_key=api_key, base_url=base_url, jitter_seed=seed)
        return OpenAIClient(model=model, api_key=api_key, jitter_seed=seed)
    if adapter == "anthropic":
        from lre.models.anthropic_api import AnthropicClient

        env = api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env)
        if not api_key:
            msg = f"Set {env} before running --adapter anthropic"
            raise click.UsageError(msg)
        return AnthropicClient(model=model, api_key=api_key, jitter_seed=seed)
    msg = f"Unknown adapter: {adapter}"  # pragma: no cover — choice already restricts
    raise click.UsageError(msg)


def _build_llm_judge(
    api_key_env: str | None,
    seed: int,
    *,
    flag_name: str = "--judge",
) -> LLMJudge:
    """Build an Anthropic-backed LLMJudge from env config.

    ``flag_name`` is the CLI flag that triggered the call — used in the
    error message when the API key is missing so the operator sees the
    exact flag they passed (``--judge`` vs. ``--outer-judge`` for the
    paired-defense command).
    """
    from lre.models.anthropic_api import AnthropicClient

    env = api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env)
    if not api_key:
        msg = f"Set {env} before running {flag_name} llm"
        raise click.UsageError(msg)
    judge_client = AnthropicClient(model="claude-3-5-sonnet-latest", api_key=api_key)
    return LLMJudge(client=judge_client, seed=seed)


@main.command(help="Run a model adapter against one or more suites.")
@click.option("--model", required=True, help="Identifier used in the report.")
@click.option(
    "--suite",
    "suites",
    multiple=True,
    required=True,
    help="Suite name; repeat for multiple suites.",
)
@click.option("--seed", type=click.IntRange(min=0), default=42, show_default=True)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0, max=5.0),
    default=0.0,
    show_default=True,
    help=(
        "Sampling temperature. Upper bound is 5.0 because some local "
        "HF checkpoints and OpenAI-compatible endpoints (vLLM, Azure "
        "deployments, OpenRouter) accept temperatures above the OpenAI "
        "public-API ceiling of 2.0. The hard cap rejects nonsense values "
        "(e.g. 10.0) without locking out legitimate experiments."
    ),
)
@click.option("--max-tokens", type=click.IntRange(min=1), default=512, show_default=True)
@click.option(
    "--judge",
    type=click.Choice(["rule", "llm"]),
    default="rule",
    show_default=True,
)
@click.option(
    "--adapter",
    type=click.Choice(["fake", "hf", "openai", "anthropic"]),
    default="fake",
    show_default=True,
    help=(
        "Adapter to instantiate. 'fake' uses the deterministic synthetic "
        "client. 'hf' loads a local HuggingFace model (requires --model-id "
        "and the 'hf' extra). 'openai' / 'anthropic' call public APIs "
        "and require the matching API key in the environment."
    ),
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="HF Hub repo id or local path (only used by --adapter hf).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="Override base URL (only used by --adapter openai).",
)
@click.option(
    "--api-key-env",
    type=str,
    default=None,
    help="Name of the env var holding the API key. Defaults to OPENAI_API_KEY / ANTHROPIC_API_KEY depending on adapter.",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=4,
    show_default=True,
    help="Maximum number of concurrent prompts in flight.",
)
@click.option(
    "--fake-refusal-rate",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help=(
        "Target synthetic refusal rate for the fake client. Ignored "
        "unless --adapter fake. Set this to match the ``lre demo`` value "
        "(default 0.6) when reproducing demo numbers from ``lre run``."
    ),
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Destination JSON file for the aggregated EvalResult list.",
)
@click.option(
    "--dump-raw",
    "dump_raw",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional path for the per-prompt RawResponse JSONL stream. "
        "Lets you rejudge later with ``lre judge --in <path>`` without "
        "re-running generation."
    ),
)
@click.option(
    "--cache",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory for content-addressed RawResponse cache. When set, "
        "the runner consults the cache before calling the model and "
        "writes successful generations back. Cache key includes "
        "(model, prompt, seed, temperature, max_tokens) — perfect for "
        "iterating on a judge without burning API quota."
    ),
)
@click.option(
    "--sample",
    "sample_n",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Sample N (>=1) prompts from each suite (deterministic in "
        "--seed). The output suite name is suffixed with "
        "'[sampled N/M, seed=K]' so downstream comparisons cannot "
        "confuse a sample with a full run. If N > suite size, the "
        "runner uses all prompts and emits a warning."
    ),
)
@click.option(
    "--cache-allow-nondeterministic",
    "cache_allow_nondet",
    is_flag=True,
    default=False,
    help=(
        "Suppress the warning emitted when --cache is paired with "
        "temperature>0. A non-zero temperature is stochastic, so the "
        "first write cements one realised sample and every subsequent "
        "hit replays it instead of resampling. Use only when that "
        "behaviour is exactly what you want."
    ),
)
@click.option(
    "--allow-symlinked-cache",
    "allow_symlinked_cache",
    is_flag=True,
    default=False,
    help=(
        "Opt in to following a symlinked --cache directory. By default "
        "a symlinked cache path is refused because it can silently "
        "redirect writes to an attacker- or accident-controlled "
        "location elsewhere on disk."
    ),
)
@click.option(
    "--use-chat-template/--no-use-chat-template",
    "use_chat_template",
    default=True,
    show_default=True,
    help=(
        "[--adapter hf only] Wrap each prompt in the tokenizer's chat "
        "template before generation. Required for modern instruct "
        "models (Qwen2-Instruct, Llama-3-Instruct, …); pass "
        "--no-use-chat-template when evaluating a base / pretraining "
        "checkpoint. The flag is baked into the effective client name "
        "so the response cache cannot serve a chat-template hit to a "
        "base-model run (or vice versa)."
    ),
)
def run(
    model: str,
    suites: tuple[str, ...],
    seed: int,
    temperature: float,
    max_tokens: int,
    judge: str,
    adapter: str,
    model_id: str | None,
    base_url: str | None,
    api_key_env: str | None,
    max_concurrent: int,
    fake_refusal_rate: float,
    out: Path,
    dump_raw: Path | None,
    cache_dir: Path | None,
    sample_n: int | None,
    cache_allow_nondet: bool,
    allow_symlinked_cache: bool,
    use_chat_template: bool,
) -> None:
    try:
        client = _build_client(
            adapter,
            model,
            model_id=model_id,
            base_url=base_url,
            api_key_env=api_key_env,
            seed=seed,
            fake_refusal_rate=fake_refusal_rate,
            use_chat_template=use_chat_template,
        )
    except click.UsageError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    llm_judge: LLMJudge | None = None
    if judge == "llm":
        try:
            llm_judge = _build_llm_judge(api_key_env, seed=seed)
        except click.UsageError as exc:
            click.echo(str(exc), err=True)
            sys.exit(1)

    cache: ResponseCache | None = None
    if cache_dir is not None:
        try:
            cache = _safe_response_cache(cache_dir, allow_symlinked=allow_symlinked_cache)
        except click.UsageError as exc:
            click.echo(str(exc), err=True)
            sys.exit(1)
        _warn_nondeterministic_cache(temperature, cache_allow_nondet, cache_dir)

    try:
        results, all_raw = asyncio.run(
            _run_async(
                client,
                list(suites),
                seed,
                temperature,
                max_tokens,
                judge,
                max_concurrent,
                llm_judge,
                cache=cache,
                sample_n=sample_n,
                # ``adapter`` is constrained by ``click.Choice`` upstream to one
                # of the AdapterLiteral values; the cast tells mypy what Click
                # already enforces at runtime.
                adapter=cast(AdapterLiteral, adapter),
                fake_refusal_rate=fake_refusal_rate,
                use_chat_template=use_chat_template,
                api_key_env=api_key_env,
                base_url=base_url,
            )
        )
    except SuiteNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except InvalidSuiteName as exc:
        # v0.9 (P1-10): a sandboxed suite-name reject (e.g. ``../../etc/passwd``)
        # should render as a clean usage error, not a raw traceback.
        raise click.UsageError(str(exc)) from exc
    except click.UsageError as exc:
        # Raised from inside _run_async when RunConfig validation fails,
        # or upstream from the API adapters' AuthenticationError.
        click.echo(f"Error: {exc.message}", err=True)
        sys.exit(2)
    out_is_stdout = _is_stdout_sentinel(out)
    _write_text_out(out, to_json(results))
    if out_is_stdout:
        click.echo(f"Wrote {len(results)} EvalResult rows to stdout", err=True)
    else:
        click.echo(f"Wrote {len(results)} EvalResult rows to {out}")
    if cache is not None:
        cache_stats = cache.stats()
        click.echo(
            f"Cache: {cache_stats['hits']} hits / "
            f"{cache_stats['misses']} misses / "
            f"{cache_stats['writes']} writes",
            err=out_is_stdout,
        )
    if dump_raw is not None:
        # Emit one JSON object per RawResponse, newline-delimited. Keys
        # are sorted so the stream is byte-stable across reruns. ``-``
        # routes the stream to stdout (rare but supported for parity).
        dump_to_stdout = _is_stdout_sentinel(dump_raw)
        if dump_to_stdout:
            handle: Any = sys.stdout
            close_after = False
        else:
            dump_raw.parent.mkdir(parents=True, exist_ok=True)
            handle = dump_raw.open("w", encoding="utf-8")
            close_after = True
        try:
            for raw in all_raw:
                handle.write(
                    json.dumps(
                        raw.model_dump(mode="json"),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        finally:
            if close_after:
                handle.close()
        click.echo(
            f"Wrote {len(all_raw)} RawResponse rows to {'stdout' if dump_to_stdout else dump_raw}",
            err=dump_to_stdout or out_is_stdout,
        )


async def _run_async(
    client: ModelClient,
    suites: list[str],
    seed: int,
    temperature: float,
    max_tokens: int,
    judge: str,
    max_concurrent: int,
    llm_judge: LLMJudge | None,
    *,
    cache: ResponseCache | None = None,
    sample_n: int | None = None,
    adapter: AdapterLiteral | None = None,
    fake_refusal_rate: float | None = None,
    use_chat_template: bool | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
) -> tuple[list[EvalResult], list[RawResponse]]:
    """Run every suite and return ``(results, all_raw_responses)``.

    The raw-response list is the concatenation of per-suite generations
    in order; the CLI uses it to populate ``--dump-raw``.

    Provenance is collected once for the entire CLI invocation so every
    :class:`EvalResult` row carries an identical snapshot (same git SHA,
    same timestamp). This keeps the JSON output compact even when
    several suites are run in a single ``lre run`` call.
    """
    # ``judge`` came from click.Choice(['rule', 'llm']); narrowing here is safe.
    judge_lit: Literal["rule", "llm"] = judge  # type: ignore[assignment]
    # v0.8 provenance: bake the run config + judge template fingerprint
    # into the snapshot. ``suite_hash`` is populated per-suite below
    # (different suites within one run get different snapshots).
    # v0.9: read the prompt-template hash from the actual judge instance
    # (subclasses or custom-prompt instances override the default), so a
    # custom judge's hash is correctly reflected in provenance.
    if judge_lit == "llm":
        judge_prompt_hash = getattr(
            llm_judge,
            "prompt_template_hash",
            hash_bytes(_LLM_JUDGE_PROMPT_TEMPLATE.encode("utf-8")),
        )
    else:
        judge_prompt_hash = None
    base_provenance = collect_provenance(
        seed,
        model_id=client.name,
        temperature=temperature,
        max_tokens=max_tokens,
        judge_prompt_hash=judge_prompt_hash,
        adapter=adapter,
        # Only record fake_refusal_rate when the run actually used the
        # fake adapter — recording it for real adapters would pollute
        # the snapshot with a meaningless value.
        fake_refusal_rate=fake_refusal_rate if adapter == "fake" else None,
        sample_n=sample_n,
        judge_kind=judge_lit,
        # use_chat_template only matters for the HF adapter.
        use_chat_template=use_chat_template if adapter == "hf" else None,
        api_key_env=api_key_env if adapter in {"openai", "anthropic"} else None,
        base_url=base_url if adapter == "openai" else None,
        max_concurrent=max_concurrent,
    )
    # API clients hold a shared httpx.AsyncClient — close it on exit.
    aclose = getattr(client, "aclose", None)
    async with contextlib.AsyncExitStack() as stack:
        if aclose is not None and callable(aclose):
            stack.push_async_callback(aclose)
        if llm_judge is not None:
            inner = getattr(llm_judge, "_client", None)
            inner_aclose = getattr(inner, "aclose", None)
            if inner_aclose is not None and callable(inner_aclose):
                stack.push_async_callback(inner_aclose)
        results: list[EvalResult] = []
        all_raw: list[RawResponse] = []
        for suite in suites:
            try:
                config = RunConfig(
                    model=client.name,
                    suites=[suite],
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    judge=judge_lit,
                    max_concurrent=max_concurrent,
                )
            except ValidationError as exc:
                raise click.UsageError(_humanize_validation_error(exc)) from exc
            full_prompts = load_suite(suite)
            responses = await run_eval(
                client,
                suite,
                config,
                cache=cache,
                sample_n=sample_n,
                prompts=full_prompts,
            )
            all_raw.extend(responses)
            # When sampling, the prompts the runner actually used are a
            # subset of the full suite. Recover them by prompt_id so the
            # per-category breakdown only counts the prompts that were
            # actually queried.
            if sample_n is not None and sample_n > 0:
                sampled_ids = [r.prompt_id for r in responses]
                full_by_id = {p.id: p for p in full_prompts}
                prompts = [full_by_id[pid] for pid in sampled_ids if pid in full_by_id]
            else:
                prompts = list(full_prompts)
            labels = await ajudge_responses(responses, kind=judge_lit, llm_judge=llm_judge)
            # Tag the suite name on the result with the sample marker so
            # downstream comparisons (``lre compare``) cannot conflate
            # a sample with a full run. v0.12: the suffix is also kept
            # when ``--sample`` overshoots the suite size (with a
            # ``capped`` marker) — pre-v0.12 dropped it silently
            # exactly when the operator most needs to know.
            suite_label = suite
            if sample_n is not None and sample_n > 0:
                if len(prompts) < len(full_prompts):
                    suite_label = (
                        f"{suite}[sampled {len(prompts)}/{len(full_prompts)}, seed={seed}]"
                    )
                elif sample_n > len(full_prompts):
                    suite_label = (
                        f"{suite}[sampled {len(full_prompts)}/{len(full_prompts)}, "
                        f"seed={seed}, capped]"
                    )
            # Per-suite provenance carries a suite-specific
            # ``suite_hash``. The other fields are shared across suites
            # in the same ``lre run`` invocation.
            suite_hash = suite_bytes_hash(suite)
            per_suite_provenance = base_provenance.model_copy(update={"suite_hash": suite_hash})
            results.append(
                aggregate_results(
                    model=client.name,
                    suite=suite_label,
                    responses=responses,
                    labels=labels,
                    prompts=prompts,
                    with_provenance=True,
                    provenance=per_suite_provenance,
                )
            )
        return results, all_raw


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


@main.command(help="Rejudge a previously cached list of RawResponse objects.")
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help=(
        "JSON or JSONL file containing RawResponse rows (a JSON array, "
        "or the newline-delimited stream emitted by `lre run --dump-raw`)."
    ),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Destination JSON file for the produced RefusalLabel rows.",
)
@click.option(
    "--judge",
    type=click.Choice(["rule", "llm"]),
    default="rule",
    show_default=True,
)
@click.option(
    "--api-key-env",
    type=str,
    default=None,
    help="Env var holding the LLM-judge API key. Defaults to ANTHROPIC_API_KEY.",
)
def judge(in_path: Path, out_path: Path, judge: str, api_key_env: str | None) -> None:
    # Accept either a JSON array (legacy) or a JSONL stream (the format
    # ``lre run --dump-raw`` emits). We try JSON-array first, then fall
    # back to JSONL on parse failure.
    rows = _load_raw_responses(in_path)
    try:
        responses = [RawResponse(**row) for row in rows]
    except ValidationError as exc:
        msg = f"Failed to parse raw-response rows in {in_path}: {_humanize_validation_error(exc)}"
        raise click.UsageError(msg) from exc
    llm_judge: LLMJudge | None = None
    if judge == "llm":
        try:
            llm_judge = _build_llm_judge(api_key_env, seed=0)
        except click.UsageError as exc:
            click.echo(str(exc), err=True)
            sys.exit(1)

    async def _do() -> list[RefusalLabel]:
        # ``judge`` came from click.Choice(['rule', 'llm']) so the narrowing is safe.
        kind: Literal["rule", "llm"] = judge  # type: ignore[assignment]
        try:
            return await ajudge_responses(responses, kind=kind, llm_judge=llm_judge)
        finally:
            if llm_judge is not None:
                inner = getattr(llm_judge, "_client", None)
                inner_aclose = getattr(inner, "aclose", None)
                if inner_aclose is not None and callable(inner_aclose):
                    await inner_aclose()

    labels = asyncio.run(_do())
    payload = [lbl.model_dump(mode="json") for lbl in labels]
    out_is_stdout = _is_stdout_sentinel(out_path)
    _write_text_out(
        out_path,
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    if out_is_stdout:
        click.echo(f"Wrote {len(labels)} labels to stdout", err=True)
    else:
        click.echo(f"Wrote {len(labels)} labels to {out_path}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@main.command(help="Render a Markdown or JSON report from cached EvalResult rows.")
@click.argument(
    "results_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["md", "json", "scaling"]),
    default="md",
    show_default=True,
)
@click.option(
    "--diff",
    "diff_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional baseline EvalResult JSON. When supplied alongside a "
        "second labels-on-disk artifact, ``lre report --diff`` produces "
        "per-prompt deltas (refused-now-complied / complied-now-refused) "
        "instead of the rendered table."
    ),
)
@click.option(
    "--labels-current",
    "labels_current",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "RefusalLabel JSON for the CURRENT run. Required with --diff to compute per-prompt flips."
    ),
)
@click.option(
    "--labels-baseline",
    "labels_baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "RefusalLabel JSON for the BASELINE run. Required with --diff to compute per-prompt flips."
    ),
)
def report(
    results_path: Path,
    fmt: str,
    diff_path: Path | None,
    labels_current: Path | None,
    labels_baseline: Path | None,
) -> None:
    if diff_path is not None:
        # Per-prompt diff vs baseline.
        if labels_current is None or labels_baseline is None:
            msg = (
                "`--diff` requires both --labels-current and --labels-baseline "
                "(per-prompt RefusalLabel JSON files)."
            )
            raise click.UsageError(msg)
        _render_diff(
            labels_current=labels_current,
            labels_baseline=labels_baseline,
        )
        return
    try:
        results = from_json(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        msg = f"Failed to parse results file {results_path}: {exc}"
        raise click.UsageError(msg) from exc
    if fmt == "md":
        click.echo(to_markdown(results))
    elif fmt == "json":
        click.echo(to_json(results), nl=False)
    else:
        click.echo(scaling_table(results))


def _render_diff(*, labels_current: Path, labels_baseline: Path) -> None:
    """Emit a per-prompt diff between two label files.

    For every prompt id present in both files, classify the transition:

    * ``refused_now_complied`` — baseline refused, current complied.
    * ``complied_now_refused`` — baseline complied, current refused.
    * ``unchanged`` — same refusal verdict on both sides.

    Prints summary counts and the list of flipped prompt ids.
    """
    cur_rows_raw = _load_json_list(labels_current, "current labels")
    base_rows_raw = _load_json_list(labels_baseline, "baseline labels")
    try:
        cur_rows = [RefusalLabel(**row) for row in cur_rows_raw]
        base_rows = [RefusalLabel(**row) for row in base_rows_raw]
    except ValidationError as exc:
        msg = f"Failed to parse RefusalLabel rows: {_humanize_validation_error(exc)}"
        raise click.UsageError(msg) from exc
    cur_by_id = {row.prompt_id: row.refused for row in cur_rows}
    base_by_id = {row.prompt_id: row.refused for row in base_rows}
    common = sorted(set(cur_by_id) & set(base_by_id))
    refused_now_complied: list[str] = []
    complied_now_refused: list[str] = []
    unchanged = 0
    for pid in common:
        if base_by_id[pid] and not cur_by_id[pid]:
            refused_now_complied.append(pid)
        elif not base_by_id[pid] and cur_by_id[pid]:
            complied_now_refused.append(pid)
        else:
            unchanged += 1
    click.echo(f"# Per-prompt diff: {labels_baseline.name} -> {labels_current.name}")
    click.echo("")
    click.echo(f"Overlapping prompts: {len(common)}")
    click.echo(f"  unchanged:               {unchanged}")
    click.echo(f"  refused_now_complied:    {len(refused_now_complied)}")
    click.echo(f"  complied_now_refused:    {len(complied_now_refused)}")
    if refused_now_complied:
        click.echo("\n## refused_now_complied")
        for pid in refused_now_complied:
            click.echo(f"  - {pid}")
    if complied_now_refused:
        click.echo("\n## complied_now_refused")
        for pid in complied_now_refused:
            click.echo(f"  - {pid}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _sum_refusals(results: list[EvalResult]) -> tuple[int, int]:
    """Return ``(refused, judged)`` summed across every row."""
    refused = sum(r.refused for r in results)
    judged = sum(r.refused + r.complied for r in results)
    return refused, judged


def _distinct_pairs(results: list[EvalResult]) -> set[tuple[str, str]]:
    """Return the set of distinct ``(model, suite)`` pairs in ``results``."""
    return {(r.model, r.suite) for r in results}


def _validate_compare_inputs(
    rows_a: list[EvalResult],
    rows_b: list[EvalResult],
    by: str | None,
) -> None:
    """Refuse to silently aggregate across multiple suites/models.

    Without this check, ``_sum_refusals`` collapses every row regardless
    of the ``(model, suite)`` tuple — comparing a 100-prompt
    ``harmful_helpful`` run against a 50-prompt ``jailbreak_styles`` run
    silently gives a single Δ that no statistician would believe.

    The caller may opt into per-suite aggregation via ``--by suite``;
    that path skips the multi-pair check.

    Raises :class:`click.UsageError` on any violation. Always called
    before the actual comparison fires.
    """
    if by == "suite":
        return
    pairs_a = _distinct_pairs(rows_a)
    pairs_b = _distinct_pairs(rows_b)
    if len(pairs_a) > 1:
        msg = (
            f"Cannot compare: file A contains {len(pairs_a)} distinct "
            f"(model, suite) pairs. Filter to a single pair first, or use "
            f"`lre compare --by suite` for per-suite comparison."
        )
        raise click.UsageError(msg)
    if len(pairs_b) > 1:
        msg = (
            f"Cannot compare: file B contains {len(pairs_b)} distinct "
            f"(model, suite) pairs. Filter to a single pair first, or use "
            f"`lre compare --by suite` for per-suite comparison."
        )
        raise click.UsageError(msg)


def _from_json_safely(path: Path) -> list[EvalResult]:
    """Parse an EvalResult JSON file with CLI-friendly errors."""
    try:
        return from_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, ValidationError, OSError) as exc:
        msg = f"Failed to parse results file {path}: {exc}"
        raise click.UsageError(msg) from exc


@main.command("compare", help="Compare two result JSON files: Δ refusal-rate + proportion test.")
@click.argument(
    "results_a",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "results_b",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--by",
    type=click.Choice(["suite"]),
    default=None,
    help=(
        "Aggregation mode. By default, files must contain exactly one "
        "(model, suite) pair each. ``--by suite`` produces a per-suite "
        "delta table across multi-row inputs."
    ),
)
@click.option(
    "--confidence",
    type=click.Choice(["0.90", "0.95", "0.99"]),
    default="0.95",
    show_default=True,
    callback=_confidence_callback,
    help=(
        "Confidence level for Wilson and Newcombe intervals. Restricted "
        "to the supported choices so an unsupported value errors with a "
        "clean Click message instead of a raw stats traceback."
    ),
)
def compare(
    results_a: Path,
    results_b: Path,
    by: str | None,
    confidence: float,
) -> None:
    a = _from_json_safely(results_a)
    b = _from_json_safely(results_b)
    if not a or not b:
        raise click.UsageError("Both result files must contain at least one row.")
    _validate_compare_inputs(a, b, by=by)
    if by == "suite":
        _compare_by_suite(a, b, results_a, results_b, confidence=confidence)
        return
    # Single-pair path. By _validate_compare_inputs we know there is
    # exactly one (model, suite) tuple per file. Surface the values in
    # the header so the user sees what they are comparing.
    pair_a = next(iter(_distinct_pairs(a)))
    pair_b = next(iter(_distinct_pairs(b)))
    suite_warning = ""
    if pair_a[1] != pair_b[1]:
        suite_warning = (
            f"WARNING: comparing different suites (A: {pair_a[1]!r}, "
            f"B: {pair_b[1]!r}). Refusal rates are not directly comparable."
        )
        click.echo(suite_warning, err=True)
    ref_a, n_a = _sum_refusals(a)
    ref_b, n_b = _sum_refusals(b)
    if n_a == 0 or n_b == 0:
        raise click.UsageError("One of the inputs has zero judged prompts; cannot compute Δ.")
    stats = compute_proportion_diff_test(ref_a, n_a, ref_b, n_b, confidence=confidence)
    p_a = ref_a / n_a
    p_b = ref_b / n_b
    low_a, high_a = compute_wilson_ci(ref_a, n_a, confidence=confidence)
    low_b, high_b = compute_wilson_ci(ref_b, n_b, confidence=confidence)
    verdict = (
        "B refuses MORE than A"
        if stats["delta"] > 0 and stats["p_value"] < 0.05
        else (
            "B refuses LESS than A"
            if stats["delta"] < 0 and stats["p_value"] < 0.05
            else "no statistically significant difference at alpha=0.05"
        )
    )
    ci_pct = f"{round(confidence * 100)}%"
    lines: list[str] = [
        f"# Refusal-rate comparison: {results_a.name} vs {results_b.name}",
        "",
        f"A: model={pair_a[0]!r} suite={pair_a[1]!r}",
        f"B: model={pair_b[0]!r} suite={pair_b[1]!r}",
        "",
        f"| Source | Refused | Judged | Refusal rate | {ci_pct} Wilson CI |",
        "|---|---|---|---|---|",
        f"| {results_a.name} | {ref_a} | {n_a} | {p_a:.4f} | [{low_a:.4f}, {high_a:.4f}] |",
        f"| {results_b.name} | {ref_b} | {n_b} | {p_b:.4f} | [{low_b:.4f}, {high_b:.4f}] |",
        "",
        f"| Delta (B - A) | {ci_pct} CI on Delta | p-value (two-sided) |",
        "|---|---|---|",
        (
            f"| {stats['delta']:+.4f} | "
            f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] | "
            f"{stats['p_value']:.4f} |"
        ),
        "",
        f"Verdict: {verdict}.",
        "",
    ]
    click.echo("\n".join(lines))


def _compare_by_suite(
    a: list[EvalResult],
    b: list[EvalResult],
    path_a: Path,
    path_b: Path,
    *,
    confidence: float,
) -> None:
    """``--by suite``: per-suite Δ refusal-rate table.

    For every suite that appears in BOTH files, sum refusals across all
    rows in each file (which may include multiple models per suite). The
    output is one table row per shared suite.
    """
    suites = sorted({r.suite for r in a} & {r.suite for r in b})
    if not suites:
        raise click.UsageError(
            "No suites are shared between the two files; cannot aggregate by suite."
        )
    ci_pct = f"{round(confidence * 100)}%"
    lines: list[str] = [
        f"# Per-suite refusal-rate comparison: {path_a.name} vs {path_b.name}",
        "",
        (
            f"| Suite | A refused | A judged | B refused | B judged | "
            f"Delta (B - A) | {ci_pct} CI on Delta | p-value |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for suite in suites:
        a_rows = [r for r in a if r.suite == suite]
        b_rows = [r for r in b if r.suite == suite]
        ref_a, n_a = _sum_refusals(a_rows)
        ref_b, n_b = _sum_refusals(b_rows)
        if n_a == 0 or n_b == 0:
            lines.append(f"| {suite} | {ref_a} | {n_a} | {ref_b} | {n_b} | — | — | — |")
            continue
        stats = compute_proportion_diff_test(ref_a, n_a, ref_b, n_b, confidence=confidence)
        lines.append(
            f"| {suite} | {ref_a} | {n_a} | {ref_b} | {n_b} | "
            f"{stats['delta']:+.4f} | "
            f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] | "
            f"{stats['p_value']:.4f} |"
        )
    lines.append("")
    click.echo("\n".join(lines))


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


@main.command("lint", help="Validate a suite JSONL file (categories, dup ids, etc.).")
@click.argument(
    "suite_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def lint(suite_path: Path) -> None:
    issues = lint_suite_file(suite_path)
    if not issues:
        click.echo(f"OK: {suite_path}")
        return
    for issue in issues:
        # File-level issues (no line number) drop the
        # ``:line_no:`` segment instead of rendering an awkward ``:-:``.
        if issue.line_no is None:
            click.echo(f"{suite_path}: {issue.severity}: {issue.message}", err=True)
        else:
            click.echo(
                f"{suite_path}:{issue.line_no}: {issue.severity}: {issue.message}",
                err=True,
            )
    sys.exit(1)


# ---------------------------------------------------------------------------
# reproduce
# ---------------------------------------------------------------------------


@main.command(
    "reproduce",
    help="Reconstruct the lre run invocation that produced a results JSON file.",
)
@click.argument(
    "results_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional destination for the reconstructed JSON when --exec is set. "
        "Pass `-` to write to stdout. Defaults to a sibling of the input "
        "named ``<basename>.reproduced.json``."
    ),
)
@click.option(
    "--exec",
    "do_exec",
    is_flag=True,
    default=False,
    help=(
        "Re-run the reconstructed invocation in-process and write the "
        "result to --out. With SOURCE_DATE_EPOCH pinned to the original "
        "run's epoch, the output matches the input byte-for-byte. "
        "For ``adapter=fake`` the rerun is fully local. For "
        "``adapter=hf`` / ``openai`` / ``anthropic`` the rerun issues "
        "real API calls or loads the real model; set the relevant env "
        "vars first."
    ),
)
def reproduce(results_path: Path, out_path: Path | None, do_exec: bool) -> None:
    """Print (or run) the ``lre run`` invocation a results file was produced from.

    Every input row carries a :class:`~lre.provenance.Provenance` snapshot
    capturing the full set of CLI inputs needed to rebuild the original
    invocation: ``adapter``, ``model_id``, ``seed``, ``temperature``,
    ``max_tokens``, ``max_concurrent``, ``judge_kind``,
    ``fake_refusal_rate``, ``sample_n``, ``use_chat_template``,
    ``api_key_env``, ``base_url``. The reproduce command unpacks those
    fields into a flat ``lre run`` command line, one per group, plus a
    header comment block that pins the original git SHA, lre version, and
    timestamp. With ``--exec`` it executes the reconstructed invocation
    in-process and writes the output to ``--out`` — the new JSON matches
    the input byte-for-byte (modulo ``provenance.run_timestamp_utc``)
    when ``SOURCE_DATE_EPOCH`` is set to the original timestamp.

    v1.0 schema is required. Pre-v1.0 results files lack the
    ``adapter`` field and the reproduce command refuses to guess —
    re-run with v0.10+ to capture the full provenance.
    """
    try:
        results = from_json(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        msg = f"Failed to parse results file {results_path}: {exc}"
        raise click.UsageError(msg) from exc
    if not results:
        raise click.UsageError(f"{results_path} contains no EvalResult rows")

    # Strip ``[sampled N/M, seed=...]`` decoration so the reconstructed
    # ``--suite`` value matches the original suite name. The sample size
    # is recovered from ``provenance.sample_n`` rather than the suffix.
    def _bare_suite(suite_label: str) -> str:
        return suite_label.split("[", 1)[0]

    # Group rows by every CLI input that drives the eval. Pre-v0.10
    # grouped by 4-tuple (seed, model_id, temperature, max_tokens) which
    # silently collapsed runs that differed only in judge / adapter /
    # sample_n. v0.10 includes the full set so two runs only collapse
    # when they were genuinely launched with the same flags.
    GroupKey = tuple[
        int,  # seed
        str | None,  # model_id
        float | None,  # temperature
        int | None,  # max_tokens
        str | None,  # adapter
        str | None,  # judge_kind
        float | None,  # fake_refusal_rate
        int | None,  # sample_n
        bool | None,  # use_chat_template
        str | None,  # api_key_env
        str | None,  # base_url
        int | None,  # max_concurrent
    ]
    groups: dict[GroupKey, list[EvalResult]] = {}
    for row in results:
        prov = row.provenance
        if prov is None:
            raise click.UsageError(
                f"row for model={row.model!r} suite={row.suite!r} has no provenance; "
                "cannot reproduce (re-run with v0.5+ to capture provenance)."
            )
        # v0.10 requires ``adapter`` to rebuild the right client. Pre-v1.0
        # results files (schema 0.7/0.8/0.9) have ``adapter=None``.
        if prov.adapter is None:
            raise click.UsageError(
                f"row for model={row.model!r} suite={row.suite!r} has provenance "
                f"schema_version={prov.schema_version!r}, which lacks the "
                "'adapter' field required by `lre reproduce` (v1.0+). "
                "Re-run with v0.10+ to capture full provenance, or copy the "
                "original CLI invocation manually."
            )
        key: GroupKey = (
            prov.seed,
            prov.model_id,
            prov.temperature,
            prov.max_tokens,
            prov.adapter,
            prov.judge_kind,
            prov.fake_refusal_rate,
            prov.sample_n,
            prov.use_chat_template,
            prov.api_key_env,
            prov.base_url,
            prov.max_concurrent,
        )
        groups.setdefault(key, []).append(row)

    # Header: emit metadata about the original run as a comment block.
    first_prov = results[0].provenance
    assert first_prov is not None  # validated above
    header_lines = [
        "# Reconstructed lre run invocations.",
        f"# Source results file: {results_path}",
        f"# Original lre version: {first_prov.lre_version}",
        f"# Original schema version: {first_prov.schema_version}",
        f"# Original run timestamp: {first_prov.run_timestamp_utc}",
        f"# Original git sha: {first_prov.git_sha or 'unknown'}"
        + (" (dirty)" if first_prov.git_dirty else ""),
        "# To reproduce byte-identical provenance, set:",
        f"#   SOURCE_DATE_EPOCH=<unix epoch of {first_prov.run_timestamp_utc}>",
    ]

    command_lines: list[str] = []
    for key, rows in groups.items():
        (
            seed,
            model_id,
            temperature,
            max_tokens,
            adapter,
            judge_kind,
            fake_refusal_rate,
            sample_n,
            use_chat_template,
            api_key_env,
            base_url,
            max_concurrent,
        ) = key
        suite_args = " ".join(f"--suite {_bare_suite(r.suite)}" for r in rows)
        # ``--model`` is the reporter identifier; the bare model_id is the
        # natural value to round-trip. ``--model-id`` is only meaningful
        # for ``--adapter hf`` but the original CLI always required
        # ``--model`` so we always emit it.
        parts = [
            "lre run",
            f"--adapter {adapter}",
            f"--model {model_id or 'UNKNOWN'}",
        ]
        if adapter == "hf" and model_id:
            parts.append(f"--model-id {model_id}")
        parts.append(suite_args)
        parts.append(f"--seed {seed}")
        if temperature is not None:
            parts.append(f"--temperature {temperature}")
        if max_tokens is not None:
            parts.append(f"--max-tokens {max_tokens}")
        if max_concurrent is not None:
            parts.append(f"--max-concurrent {max_concurrent}")
        if judge_kind is not None:
            parts.append(f"--judge {judge_kind}")
        if adapter == "fake" and fake_refusal_rate is not None:
            parts.append(f"--fake-refusal-rate {fake_refusal_rate}")
        if sample_n is not None:
            parts.append(f"--sample {sample_n}")
        if adapter == "hf" and use_chat_template is not None:
            parts.append("--use-chat-template" if use_chat_template else "--no-use-chat-template")
        if api_key_env:
            parts.append(f"--api-key-env {api_key_env}")
        if base_url:
            parts.append(f"--base-url {base_url}")
        parts.append("--out result.json")
        command_lines.append(" ".join(parts))

    if not do_exec:
        for line in header_lines:
            click.echo(line)
        for cmd in command_lines:
            click.echo(cmd)
        return

    # --exec path: re-run each group in-process. For ``adapter=fake`` the
    # rerun is fully local and side-effect-free. For real adapters we
    # rebuild the original client (HF / OpenAI / Anthropic) — that path
    # issues real API calls or loads real models and is documented in
    # the --exec help text.
    new_results: list[EvalResult] = []
    for key, rows in groups.items():
        (
            seed,
            model_id,
            temperature,
            max_tokens,
            adapter,
            judge_kind,
            fake_refusal_rate,
            sample_n,
            use_chat_template,
            api_key_env,
            base_url,
            max_concurrent,
        ) = key
        suite_names = [_bare_suite(r.suite) for r in rows]
        # Build the client matching the original adapter. The fake
        # adapter is the common case for "validate the harness is
        # deterministic"; real adapters re-issue real work and require
        # the operator to have set the relevant env vars.
        #
        # v0.11: For the HF adapter, ``Provenance.model_id`` is set to
        # ``client.name`` which carries a ``@chat`` suffix when
        # ``use_chat_template=True``. The raw HF Hub repo id (the
        # value passed to ``HFLocalClient(model_id=...)``) is the part
        # before the suffix. Strip ``@chat`` so the reconstructed
        # client actually loads the right repo instead of trying to
        # ``from_pretrained("Qwen/Qwen2-0.5B-Instruct@chat")`` which
        # 404s on the Hub. Other adapters keep model_id as-is.
        hf_repo_id = model_id
        if adapter == "hf" and hf_repo_id and hf_repo_id.endswith("@chat"):
            hf_repo_id = hf_repo_id[: -len("@chat")]
        try:
            client = _build_client(
                adapter or "fake",
                model_id or "fake-1b",
                model_id=hf_repo_id,
                base_url=base_url,
                api_key_env=api_key_env,
                seed=seed,
                fake_refusal_rate=fake_refusal_rate if fake_refusal_rate is not None else 0.5,
                use_chat_template=use_chat_template if use_chat_template is not None else True,
            )
        except click.UsageError as exc:
            click.echo(str(exc), err=True)
            sys.exit(1)
        # Build the matching judge. ``judge_kind`` defaults to "rule"
        # for any pre-v1.0 row that slipped through (shouldn't happen
        # given the adapter guard above, but defensive).
        llm_judge: LLMJudge | None = None
        if judge_kind == "llm":
            try:
                llm_judge = _build_llm_judge(api_key_env, seed=seed)
            except click.UsageError as exc:
                click.echo(str(exc), err=True)
                sys.exit(1)
        repro_results, _raw = asyncio.run(
            _run_async(
                client,
                suite_names,
                seed,
                temperature if temperature is not None else 0.0,
                max_tokens if max_tokens is not None else 512,
                judge_kind or "rule",
                max_concurrent if max_concurrent is not None else 4,
                llm_judge,
                cache=None,
                sample_n=sample_n,
                # ``adapter`` came from ``Provenance.adapter`` which is
                # validated as an AdapterLiteral at construction time.
                adapter=adapter,  # type: ignore[arg-type]
                fake_refusal_rate=fake_refusal_rate,
                use_chat_template=use_chat_template,
                api_key_env=api_key_env,
                base_url=base_url,
            )
        )
        new_results.extend(repro_results)
    # Resolve the destination.
    if out_path is None:
        out_path = results_path.with_suffix(".reproduced.json")
    out_is_stdout = _is_stdout_sentinel(out_path)
    _write_text_out(out_path, to_json(new_results))
    if out_is_stdout:
        click.echo(
            f"Re-ran {len(new_results)} EvalResult rows; wrote to stdout.",
            err=True,
        )
    else:
        click.echo(f"Re-ran {len(new_results)} EvalResult rows; wrote to {out_path}.")


# ---------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------


@main.command(
    "kappa",
    help="Cohen's κ inter-judge agreement on two RefusalLabel JSON files.",
)
@click.option(
    "--judge-a",
    "judge_a",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON list of RefusalLabel rows from judge A.",
)
@click.option(
    "--judge-b",
    "judge_b",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON list of RefusalLabel rows from judge B.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help=(
        "Exit non-zero if either file contains prompt_ids missing from "
        "the other. Without --strict, non-overlapping ids are dropped "
        "with a warning."
    ),
)
def kappa(judge_a: Path, judge_b: Path, strict: bool) -> None:
    raw_a = _load_json_list(judge_a, "labels A")
    raw_b = _load_json_list(judge_b, "labels B")
    try:
        rows_a = [RefusalLabel(**row) for row in raw_a]
        rows_b = [RefusalLabel(**row) for row in raw_b]
    except ValidationError as exc:
        msg = f"Failed to parse RefusalLabel rows: {_humanize_validation_error(exc)}"
        raise click.UsageError(msg) from exc
    by_id_a = {row.prompt_id: row.refused for row in rows_a}
    by_id_b = {row.prompt_id: row.refused for row in rows_b}
    common_set = set(by_id_a) & set(by_id_b)
    only_a = sorted(set(by_id_a) - set(by_id_b))
    only_b = sorted(set(by_id_b) - set(by_id_a))
    # Tell the user what was dropped. Silent intersection
    # hides config bugs (mismatched judge runs against different prompt
    # sets); surface the drop with sample ids and a top-level warning.
    if only_a:
        sample = ", ".join(repr(p) for p in only_a[:3])
        click.echo(
            f"Dropped {len(only_a)} rows present only in A (e.g., {sample})",
            err=True,
        )
    if only_b:
        sample = ", ".join(repr(p) for p in only_b[:3])
        click.echo(
            f"Dropped {len(only_b)} rows present only in B (e.g., {sample})",
            err=True,
        )
    if (only_a or only_b) and strict:
        msg = (
            "Non-overlapping prompt_ids detected and --strict was set; "
            "refusing to compute κ over a partial intersection."
        )
        raise click.UsageError(msg)
    if only_a or only_b:
        click.echo(
            "WARNING: non-overlapping prompt_ids — κ computed over the overlapping subset only.",
            err=True,
        )
    if not common_set:
        raise click.UsageError("No overlapping prompt_ids between the two files.")
    common = sorted(common_set)
    labels_a = [by_id_a[pid] for pid in common]
    labels_b = [by_id_b[pid] for pid in common]
    k = cohen_kappa(labels_a, labels_b)
    click.echo(f"Cohen's κ (n={len(common)} overlapping prompts): {k:.4f}")


# ---------------------------------------------------------------------------
# defense-in-depth (`lre did`)
# ---------------------------------------------------------------------------


_DID_HELP = (
    "Compute paired-defense ('defense-in-depth') refusal stats for an "
    "(inner, outer) judge pipeline. Pairs a model-side judge with an "
    "outer classifier and reports inner / outer / joint refusal rates. "
    "Inspired by FAR.AI's research on layered LLM defenses (see "
    "https://www.far.ai/news, including the STACK pipeline-attacks "
    "work) which finds that stacking a refusal classifier on top of a "
    "model leaves measurable vulnerabilities an adaptive attacker can "
    "exploit. This command is the measurement side of that finding: "
    "use it to quantify how much (or how little) the outer classifier "
    "actually shifts the joint refusal rate versus the model-only "
    "baseline on your prompt suite."
)


@main.command("did", help=_DID_HELP)
@click.option(
    "--responses",
    "responses_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Raw responses (JSON array or JSONL stream from `lre run --dump-raw`).",
)
@click.option(
    "--inner-judge",
    type=click.Choice(["rule", "llm"]),
    default="rule",
    show_default=True,
    help="Inner (model-side) judge.",
)
@click.option(
    "--outer-judge",
    type=click.Choice(["rule", "llm"]),
    default="llm",
    show_default=True,
    help="Outer (classifier-side) judge.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help=(
        "Destination file for the aggregated paired-defense report. "
        "Optional when --format md is used — the table prints to stdout."
    ),
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["md", "json"]),
    default="md",
    show_default=True,
    help=(
        "Output format. 'md' prints a Markdown table to stdout (and "
        "writes it to --out when supplied). 'json' writes the full "
        "report (summary + per-prompt rows) to --out, which becomes "
        "required."
    ),
)
@click.option(
    "--api-key-env",
    type=str,
    default=None,
    help="Env var holding the LLM-judge API key. Defaults to ANTHROPIC_API_KEY.",
)
def did(
    responses_path: Path,
    inner_judge: str,
    outer_judge: str,
    out_path: Path | None,
    fmt: str,
    api_key_env: str | None,
) -> None:
    if fmt == "json" and out_path is None:
        msg = "--format json requires --out <path>"
        raise click.UsageError(msg)
    if inner_judge == outer_judge:
        # Layered defense requires orthogonal classifiers; identical
        # judges produce identical labels so the joint rate degenerates
        # to the inner rate. Warn but continue so smoke tests still work.
        click.echo(
            f"WARNING: inner and outer judges are both {inner_judge!r} — "
            "defense-in-depth requires orthogonal classifiers to be "
            "meaningful. Consider --outer-judge llm.",
            err=True,
        )
    rows = _load_raw_responses(responses_path)
    try:
        responses = [RawResponse(**row) for row in rows]
    except ValidationError as exc:
        msg = (
            f"Failed to parse raw-response rows in {responses_path}: "
            f"{_humanize_validation_error(exc)}"
        )
        raise click.UsageError(msg) from exc

    inner = _resolve_did_judge(inner_judge, api_key_env, flag_name="--inner-judge")
    outer = _resolve_did_judge(outer_judge, api_key_env, flag_name="--outer-judge")

    async def _run() -> dict[str, Any]:
        try:
            defense = PairedDefense(inner_judge=inner, outer_judge=outer)
            inner_labels, outer_labels, joint, ambiguous = await paired_label_responses(
                defense, responses
            )
            stats = aggregate_paired_results(inner_labels, outer_labels, joint, ambiguous)
            per_prompt = [
                {
                    "prompt_id": il.prompt_id,
                    "model": il.model,
                    "inner_refused": il.refused,
                    "outer_refused": ol.refused,
                    "system_refused": j,
                }
                for il, ol, j in zip(inner_labels, outer_labels, joint, strict=True)
            ]
            return {"summary": stats, "per_prompt": per_prompt}
        finally:
            for judge_obj in (inner, outer):
                inner_client = getattr(judge_obj, "_client", None)
                inner_aclose = getattr(inner_client, "aclose", None)
                if inner_aclose is not None and callable(inner_aclose):
                    await inner_aclose()

    report = asyncio.run(_run())
    summary = report["summary"]
    if fmt == "md":
        markdown = to_markdown_did(summary)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
        click.echo(markdown)
        return
    # fmt == "json"
    assert out_path is not None  # narrowed by the early validation
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    click.echo(
        f"Defense-in-depth: inner {summary['inner_refusal_rate']:.3f} | "
        f"outer {summary['outer_refusal_rate']:.3f} | "
        f"joint {summary['joint_refusal_rate']:.3f} "
        f"(Δ {summary['delta_vs_inner_only']:+.3f} vs. inner-only). "
        f"Wrote report to {out_path}."
    )


def _resolve_did_judge(kind: str, api_key_env: str | None, *, flag_name: str) -> Any:
    """Build a judge instance from a CLI ``--inner-judge`` / ``--outer-judge`` choice.

    ``flag_name`` is forwarded into the missing-API-key error message so
    the operator sees the exact flag that triggered the failure.
    """
    if kind == "rule":
        return RuleBasedJudge()
    if kind == "llm":
        return _build_llm_judge(api_key_env, seed=0, flag_name=flag_name)
    msg = f"unknown judge kind {kind!r}"  # pragma: no cover - click.Choice guards
    raise click.UsageError(msg)


# ---------------------------------------------------------------------------
# cache (info / clear)
# ---------------------------------------------------------------------------


@main.group(help="Inspect and prune the on-disk RawResponse cache.")
def cache() -> None:
    """Manage the content-addressed cache populated by ``lre run --cache``."""


def _iter_cache_files(cache_dir: Path) -> list[Path]:
    """Return every ``*.json`` cache entry under ``cache_dir``.

    Walks the sharded ``<dir>/<key[:2]>/<key>.json`` layout. Excludes:

    * ``*.tmp.json`` staging files left by a crashed atomic write — they
      are not cache entries.
    * The ``.lre-cache`` sentinel — it's metadata, not a cached row.
    """
    if not cache_dir.is_dir():
        return []
    return sorted(
        p
        for p in cache_dir.rglob("*.json")
        if p.is_file() and not p.name.endswith(".tmp.json") and p.name != SENTINEL_FILENAME
    )


def _parse_duration(spec: str) -> float:
    """Parse a duration like ``"7d"`` / ``"24h"`` / ``"30m"`` / ``"45s"`` into seconds.

    Bare integers are treated as days for symmetry with ``find -mtime``.
    Raises :class:`click.UsageError` on a malformed spec so the CLI
    surface stays clean.
    """
    text = spec.strip().lower()
    if not text:
        msg = "Empty duration"
        raise click.UsageError(msg)
    unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text[-1] in unit_seconds:
        try:
            value = float(text[:-1])
        except ValueError as exc:
            msg = f"Invalid duration {spec!r}; expected <N>[s|m|h|d]"
            raise click.UsageError(msg) from exc
        return value * unit_seconds[text[-1]]
    try:
        return float(text) * 86400  # bare number => days
    except ValueError as exc:
        msg = f"Invalid duration {spec!r}; expected <N>[s|m|h|d]"
        raise click.UsageError(msg) from exc


def _format_bytes(n: int) -> str:
    """Render a byte count in human-readable units."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


@cache.command("info", help="Summarise the cache directory: entry count, size, age range.")
@click.option(
    "--dir",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Cache directory to inspect (the same path passed to `lre run --cache`).",
)
def cache_info(cache_dir: Path) -> None:
    if not cache_dir.exists():
        click.echo(f"Cache: directory {cache_dir} does not exist.")
        return
    if not is_lre_cache_dir(cache_dir):
        # Surface a softer message when the directory exists but lacks
        # the sentinel — it might be a pre-v0.7 cache the user should
        # migrate, not a stray directory. Point at ``cache migrate``.
        msg = (
            f"Refusing to inspect {cache_dir}: missing or invalid "
            f"{SENTINEL_FILENAME!r} sentinel. If this is a pre-v0.7 "
            f"cache directory, run `lre cache migrate --dir {cache_dir}` "
            "to upgrade it; otherwise point --dir at a directory "
            "created by `lre run --cache`."
        )
        raise click.UsageError(msg)
    files = _iter_cache_files(cache_dir)
    if not files:
        click.echo(f"Cache: 0 entries in {cache_dir}.")
        return
    total_bytes = sum(p.stat().st_size for p in files)
    mtimes = [p.stat().st_mtime for p in files]
    # Use ``timezone.utc`` (not the ``datetime.UTC`` alias) for Python
    # 3.10 compatibility — the project's minimum supported Python.
    tz_utc = _dt.timezone.utc
    oldest = _dt.datetime.fromtimestamp(min(mtimes), tz=tz_utc).strftime("%Y-%m-%d")
    newest = _dt.datetime.fromtimestamp(max(mtimes), tz=tz_utc).strftime("%Y-%m-%d")
    click.echo(
        f"Cache: {len(files)} entries, {_format_bytes(total_bytes)}, "
        f"oldest {oldest}, newest {newest} (dir: {cache_dir})"
    )


@cache.command("clear", help="Delete cache entries, optionally restricted by age.")
@click.option(
    "--dir",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Cache directory to prune.",
)
@click.option(
    "--older-than",
    "older_than",
    type=str,
    default=None,
    help=(
        "Only delete entries whose mtime is older than this duration "
        "(e.g. '7d', '24h', '30m', '45s'; a bare number is interpreted "
        "as days). Omit to delete every entry."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List the files that would be removed without deleting anything.",
)
def cache_clear(cache_dir: Path, older_than: str | None, dry_run: bool) -> None:
    if not cache_dir.exists():
        click.echo(f"Cache: directory {cache_dir} does not exist.")
        return
    if not is_lre_cache_dir(cache_dir):
        msg = (
            f"Refusing to clear {cache_dir}: missing the "
            f"{SENTINEL_FILENAME!r} sentinel file. Point --dir at a "
            "directory created by `lre run --cache` to avoid deleting "
            "unrelated *.json files."
        )
        raise click.UsageError(msg)
    files = _iter_cache_files(cache_dir)
    if not files:
        click.echo(f"Cache: nothing to remove in {cache_dir}.")
        return
    targets: list[Path]
    if older_than is not None:
        cutoff = _time.time() - _parse_duration(older_than)
        targets = [p for p in files if p.stat().st_mtime < cutoff]
    else:
        targets = list(files)
    if not targets:
        click.echo(f"Cache: 0 of {len(files)} entries match the filter; nothing to remove.")
        return
    if dry_run:
        click.echo(
            f"Cache: would remove {len(targets)} of {len(files)} entries from {cache_dir} "
            "(dry-run; no files were modified)."
        )
        return
    removed = 0
    for path in targets:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            click.echo(f"Cache: failed to remove {path}: {exc}", err=True)
    click.echo(f"Cache: removed {removed} of {len(files)} entries from {cache_dir}.")


@cache.command("migrate", help="Upgrade a pre-v0.7 cache directory to the v0.7+ layout.")
@click.option(
    "--dir",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Cache directory to migrate.",
)
@click.option(
    "--purge-stale",
    is_flag=True,
    default=False,
    help=(
        "Delete cache entries whose JSON no longer parses as a "
        "RawResponse (schema drift from an older harness version). "
        "Without this flag the migrate command only writes the missing "
        ".lre-cache sentinel."
    ),
)
def cache_migrate(cache_dir: Path, purge_stale: bool) -> None:
    """Write the ``.lre-cache`` sentinel on a pre-v0.7 cache directory.

    Idempotent: running ``cache migrate`` against an already-migrated
    cache is a no-op (the sentinel is left in place and the file count
    is reported). When ``--purge-stale`` is supplied, every ``*.json``
    entry under the directory is parsed as a :class:`RawResponse`; rows
    that fail validation are deleted so the cache cannot serve a stale
    schema to a v0.8+ harness.
    """
    from lre.state import RawResponse

    if not cache_dir.is_dir():
        msg = f"--dir {cache_dir} is not a directory or does not exist."
        raise click.UsageError(msg)
    # 1) Write the sentinel if missing. Constructing a ResponseCache is
    # the canonical way — it writes the sentinel atomically.
    had_sentinel = is_lre_cache_dir(cache_dir)
    try:
        ResponseCache(cache_dir)
    except OSError as exc:
        msg = f"Cannot write sentinel to {cache_dir}: {exc}"
        raise click.UsageError(msg) from exc
    if had_sentinel:
        click.echo(f"Cache: sentinel already present at {cache_dir}.")
    else:
        click.echo(f"Cache: wrote {SENTINEL_FILENAME!r} sentinel to {cache_dir}.")
    if not purge_stale:
        return
    # 2) Walk every *.json (skipping the staged *.tmp.json artefacts and
    # the sentinel itself) and validate.
    files = _iter_cache_files(cache_dir)
    purged = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            RawResponse(**payload)
        except (OSError, json.JSONDecodeError, TypeError, ValidationError):
            try:
                path.unlink()
                purged += 1
            except OSError as exc:
                click.echo(f"Cache: failed to remove stale entry {path}: {exc}", err=True)
    click.echo(
        f"Cache: purged {purged} stale entries (of {len(files)} *.json files in {cache_dir})."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
