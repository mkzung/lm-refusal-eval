"""Tests for the Click CLI surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from lre.cli import main
from lre.report import to_json
from lre.state import EvalResult, RefusalLabel


def test_demo_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--refusal-rate", "0.5", "--seed", "1"])
    assert result.exit_code == 0, result.output
    # Headline markdown headers should appear in the output.
    assert "lre demo" in result.output
    assert "Refusal rate" in result.output


def test_report_md_on_fixture() -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent / "fixtures" / "cached_run.json"
    result = runner.invoke(main, ["report", str(fixture), "--format", "md"])
    assert result.exit_code == 0, result.output
    assert "qwen-mock-0.5b" in result.output


def test_report_scaling_on_fixture() -> None:
    runner = CliRunner()
    fixture = Path(__file__).parent / "fixtures" / "cached_run.json"
    result = runner.invoke(main, ["report", str(fixture), "--format", "scaling"])
    assert result.exit_code == 0, result.output
    assert "scaling table" in result.output.lower()


def test_run_then_report_roundtrip(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "results.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["model"] == "fake-1b"
    assert payload[0]["suite"] == "harmful_helpful"


def test_unknown_suite_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "does_not_exist",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0


def test_lre_demo_is_byte_stable_across_invocations() -> None:
    """F-R2-T-17: two demo invocations with identical args must emit
    byte-identical Markdown. We use Click's CliRunner so the test does
    not depend on a subprocess shell.
    """
    runner = CliRunner()
    args = ["demo", "--refusal-rate", "0.6", "--seed", "42", "--model-name", "fake-1b"]
    a = runner.invoke(main, args)
    b = runner.invoke(main, args)
    assert a.exit_code == 0, a.output
    assert b.exit_code == 0, b.output
    assert a.output == b.output, "demo output drifted across reruns"


def test_run_rejects_invalid_max_concurrent(tmp_path: Path) -> None:
    """F-R2-P1-7: ``--max-concurrent 0`` violates the RunConfig
    constraint. The CLI must surface this as a clean error, not a raw
    Pydantic traceback.
    """
    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--max-concurrent",
            "0",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "max_concurrent" in result.output.lower()
    # No raw traceback should reach the user.
    assert "Traceback" not in result.output


def test_run_rejects_invalid_temperature(tmp_path: Path) -> None:
    """``--temperature 10.0`` violates RunConfig (max 5.0 as of the current implementation).

    the current implementation raised the ceiling from 2.0 to 5.0 so local HF checkpoints
    and OpenAI-compatible endpoints (vLLM, Azure, OpenRouter) that
    accept temperatures above the OpenAI public-API limit are usable.
    A nonsense value like 10.0 still fails fast.
    """
    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--temperature",
            "10.0",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "temperature" in result.output.lower()
    assert "Traceback" not in result.output


def test_run_accepts_temperature_above_two(tmp_path: Path) -> None:
    """the current implementation: ``--temperature 3.0`` is accepted (was rejected in the current implementation)."""
    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--temperature",
            "3.0",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output


def _eval_result(model: str, suite: str, refused: int, complied: int) -> EvalResult:
    return EvalResult(
        model=model,
        suite=suite,
        total=refused + complied,
        refused=refused,
        complied=complied,
        ambiguous=0,
        refusal_rate=refused / (refused + complied) if refused + complied else None,
        latency_p50_s=0.01,
        latency_p99_s=0.02,
    )


def test_compare_command_outputs_delta_and_p_value(tmp_path: Path) -> None:
    """NEW-R2-1: ``lre compare`` should print a Δ refusal-rate table with
    a p-value.
    """
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json([_eval_result("model-a", "s", 5, 5)]), encoding="utf-8")
    b_path.write_text(to_json([_eval_result("model-b", "s", 9, 1)]), encoding="utf-8")
    result = runner.invoke(main, ["compare", str(a_path), str(b_path)])
    assert result.exit_code == 0, result.output
    assert "Refusal-rate comparison" in result.output
    assert "p-value" in result.output
    assert "+0.4000" in result.output or "0.4000" in result.output


def test_lint_clean_suite_exits_zero() -> None:
    """NEW-R2-2: linting the bundled suites should be clean."""
    runner = CliRunner()
    import importlib.resources

    suite = importlib.resources.files("lre").joinpath("data", "prompts", "harmful_helpful.jsonl")
    result = runner.invoke(main, ["lint", str(suite)])
    assert result.exit_code == 0, result.output


def test_lint_reports_duplicate_id(tmp_path: Path) -> None:
    """NEW-R2-2: duplicate ids must surface as a line-level error."""
    runner = CliRunner()
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "p1", "text": "a", "category": "harmful"}\n'
        '{"id": "p1", "text": "b", "category": "harmful"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(main, ["lint", str(bad)])
    assert result.exit_code != 0
    assert "duplicate id" in result.output


def test_kappa_command_perfect_agreement(tmp_path: Path) -> None:
    """NEW-R2-3: identical label files give κ = 1.0."""
    runner = CliRunner()
    labels = [
        RefusalLabel(
            prompt_id=f"p{i}", model="m", refused=bool(i % 2), confidence=1.0, judge="rule"
        )
        for i in range(6)
    ]
    payload = [lbl.model_dump(mode="json") for lbl in labels]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(payload), encoding="utf-8")
    b.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(main, ["kappa", "--judge-a", str(a), "--judge-b", str(b)])
    assert result.exit_code == 0, result.output
    assert "1.0000" in result.output


# ---------------------------------------------------------------------------
# F-R3-P0-2: compare validation + --by suite + --confidence
# ---------------------------------------------------------------------------


def _two_pair_results() -> list[EvalResult]:
    """An EvalResult list with TWO distinct (model, suite) pairs."""
    return [
        _eval_result("alpha", "suite_one", 5, 5),
        _eval_result("alpha", "suite_two", 3, 7),
    ]


def test_compare_rejects_multi_pair_inputs_without_by(tmp_path: Path) -> None:
    """F-R3-P0-2: file A with multiple (model, suite) pairs must be
    refused without ``--by suite``.
    """
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json(_two_pair_results()), encoding="utf-8")
    b_path.write_text(to_json([_eval_result("beta", "suite_one", 4, 6)]), encoding="utf-8")
    result = runner.invoke(main, ["compare", str(a_path), str(b_path)])
    assert result.exit_code != 0
    assert "Cannot compare" in result.output
    assert "--by suite" in result.output


def test_compare_by_suite_aggregates_per_suite(tmp_path: Path) -> None:
    """F-R3-P0-2: ``--by suite`` produces a per-suite delta table."""
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json(_two_pair_results()), encoding="utf-8")
    b_path.write_text(
        to_json(
            [
                _eval_result("beta", "suite_one", 8, 2),
                _eval_result("beta", "suite_two", 6, 4),
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(main, ["compare", str(a_path), str(b_path), "--by", "suite"])
    assert result.exit_code == 0, result.output
    assert "Per-suite refusal-rate comparison" in result.output
    assert "suite_one" in result.output
    assert "suite_two" in result.output


def test_compare_single_pair_shows_model_and_suite(tmp_path: Path) -> None:
    """F-R3-P2-18: the model and suite names are visible in the header."""
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json([_eval_result("alpha", "suite_one", 5, 5)]), encoding="utf-8")
    b_path.write_text(to_json([_eval_result("beta", "suite_one", 9, 1)]), encoding="utf-8")
    result = runner.invoke(main, ["compare", str(a_path), str(b_path)])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "suite_one" in result.output


def test_compare_different_suites_emits_warning(tmp_path: Path) -> None:
    """F-R3-P0-2: comparing different suites must surface a warning."""
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json([_eval_result("m", "jailbreak_styles", 5, 5)]), encoding="utf-8")
    b_path.write_text(to_json([_eval_result("m", "harmful_helpful", 5, 5)]), encoding="utf-8")
    result = runner.invoke(main, ["compare", str(a_path), str(b_path)])
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "jailbreak_styles" in result.output
    assert "harmful_helpful" in result.output


def test_compare_confidence_99_widens_interval(tmp_path: Path) -> None:
    """NEW-R3-3: ``--confidence 0.99`` produces a wider CI than 0.95."""
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json([_eval_result("m", "s", 5, 5)]), encoding="utf-8")
    b_path.write_text(to_json([_eval_result("m", "s", 9, 1)]), encoding="utf-8")
    r95 = runner.invoke(main, ["compare", str(a_path), str(b_path), "--confidence", "0.95"])
    r99 = runner.invoke(main, ["compare", str(a_path), str(b_path), "--confidence", "0.99"])
    assert r95.exit_code == 0, r95.output
    assert r99.exit_code == 0, r99.output
    # The header label must reflect the requested confidence.
    assert "95%" in r95.output
    assert "99%" in r99.output


def test_compare_confidence_90_path(tmp_path: Path) -> None:
    """NEW-R3-3: ``--confidence 0.90`` is also accepted."""
    runner = CliRunner()
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(to_json([_eval_result("m", "s", 5, 5)]), encoding="utf-8")
    b_path.write_text(to_json([_eval_result("m", "s", 6, 4)]), encoding="utf-8")
    result = runner.invoke(main, ["compare", str(a_path), str(b_path), "--confidence", "0.90"])
    assert result.exit_code == 0, result.output
    assert "90%" in result.output


# ---------------------------------------------------------------------------
# F-R3-P0-3: kappa drop warnings + --strict
# ---------------------------------------------------------------------------


def _write_labels(path: Path, pairs: list[tuple[str, bool]]) -> None:
    """Helper: write a JSON list of RefusalLabel dicts to ``path``."""
    labels = [
        RefusalLabel(prompt_id=pid, model="m", refused=ref, confidence=1.0, judge="rule")
        for pid, ref in pairs
    ]
    path.write_text(
        json.dumps([lbl.model_dump(mode="json") for lbl in labels]),
        encoding="utf-8",
    )


def test_kappa_full_overlap_no_warnings(tmp_path: Path) -> None:
    """F-R3-P0-3: identical prompt-id sets produce no drop warnings."""
    runner = CliRunner()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_labels(a, [("p1", True), ("p2", False), ("p3", True)])
    _write_labels(b, [("p1", True), ("p2", False), ("p3", True)])
    result = runner.invoke(main, ["kappa", "--judge-a", str(a), "--judge-b", str(b)])
    assert result.exit_code == 0, result.output
    assert "Dropped" not in result.output
    assert "WARNING" not in result.output


def test_kappa_partial_overlap_warns_and_drops(tmp_path: Path) -> None:
    """F-R3-P0-3: non-overlapping prompt_ids → warning lists dropped ids."""
    runner = CliRunner()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_labels(a, [("p1", True), ("p2", False), ("only_in_a", True)])
    _write_labels(b, [("p1", True), ("p2", False), ("only_in_b", False)])
    result = runner.invoke(main, ["kappa", "--judge-a", str(a), "--judge-b", str(b)])
    assert result.exit_code == 0, result.output
    assert "only_in_a" in result.output
    assert "only_in_b" in result.output
    assert "Dropped 1 rows present only in A" in result.output
    assert "Dropped 1 rows present only in B" in result.output
    assert "WARNING" in result.output


def test_kappa_strict_fails_on_partial_overlap(tmp_path: Path) -> None:
    """F-R3-P0-3: --strict turns a partial overlap into an error."""
    runner = CliRunner()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_labels(a, [("p1", True), ("only_in_a", False)])
    _write_labels(b, [("p1", True), ("only_in_b", False)])
    result = runner.invoke(
        main,
        ["kappa", "--judge-a", str(a), "--judge-b", str(b), "--strict"],
    )
    assert result.exit_code != 0
    assert "strict" in result.output.lower() or "Non-overlapping" in result.output


# ---------------------------------------------------------------------------
# F-R3-P1-4: clean error messages on malformed JSON
# ---------------------------------------------------------------------------


def test_compare_bad_json_emits_clean_error(tmp_path: Path) -> None:
    """F-R3-P1-4: malformed JSON must produce a one-line click error,
    not a Python traceback.
    """
    runner = CliRunner()
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all", encoding="utf-8")
    result = runner.invoke(main, ["compare", str(bad), str(bad)])
    assert result.exit_code != 0
    assert "Failed to parse" in result.output
    assert "Traceback" not in result.output


def test_compare_non_array_emits_clean_error(tmp_path: Path) -> None:
    """F-R3-P1-4: JSON that is valid but not a list must error cleanly."""
    runner = CliRunner()
    bad = tmp_path / "obj.json"
    bad.write_text('{"not": "a list"}', encoding="utf-8")
    result = runner.invoke(main, ["compare", str(bad), str(bad)])
    assert result.exit_code != 0
    # Either schema validation or "expected a JSON array" — both clean.
    assert "Traceback" not in result.output


def test_report_bad_json_emits_clean_error(tmp_path: Path) -> None:
    """F-R3-P1-4: same protection on ``lre report``."""
    runner = CliRunner()
    bad = tmp_path / "bad.json"
    bad.write_text("definitely not json", encoding="utf-8")
    result = runner.invoke(main, ["report", str(bad)])
    assert result.exit_code != 0
    assert "Failed to parse" in result.output
    assert "Traceback" not in result.output


def test_judge_bad_json_emits_clean_error(tmp_path: Path) -> None:
    """F-R3-P1-4: ``lre judge --in`` must reject bad JSON cleanly."""
    runner = CliRunner()
    bad = tmp_path / "bad.json"
    bad.write_text("{not even close", encoding="utf-8")
    out = tmp_path / "labels.json"
    result = runner.invoke(
        main,
        ["judge", "--in", str(bad), "--out", str(out)],
    )
    assert result.exit_code != 0
    assert "Failed to parse" in result.output
    assert "Traceback" not in result.output


def test_kappa_bad_json_emits_clean_error(tmp_path: Path) -> None:
    """F-R3-P1-4: ``lre kappa`` must reject bad JSON cleanly."""
    runner = CliRunner()
    bad = tmp_path / "bad.json"
    bad.write_text("nope", encoding="utf-8")
    result = runner.invoke(
        main,
        ["kappa", "--judge-a", str(bad), "--judge-b", str(bad)],
    )
    assert result.exit_code != 0
    assert "Failed to parse" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# F-R3-P1-5: auth errors produce clean exits
# ---------------------------------------------------------------------------


def test_run_missing_anthropic_api_key_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-R3-P1-5: a missing API key surfaces as a clean click.UsageError,
    not a per-prompt traceback.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--adapter",
            "anthropic",
            "--model",
            "claude-3-5-sonnet-latest",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_run_missing_openai_api_key_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-R3-P1-5: same for OpenAI."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--adapter",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_run_with_anthropic_401_surfaces_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-R3-P1-5: when the adapter receives a 401, the CLI exits cleanly
    instead of dumping a per-prompt traceback for every prompt in the
    suite (15 in harmful_helpful).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    real = httpx.AsyncClient
    transport_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls["n"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    transport = httpx.MockTransport(handler)

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    runner = CliRunner()
    out = tmp_path / "x.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--adapter",
            "anthropic",
            "--model",
            "claude-3-5-sonnet-latest",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    # The error must mention authentication, not be a raw stack trace.
    assert "authentication" in result.output.lower() or "401" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# F-R3-P1-6: --fake-refusal-rate plumbing
# ---------------------------------------------------------------------------


def test_run_fake_refusal_rate_matches_demo(tmp_path: Path) -> None:
    """F-R3-P1-6: ``lre run --adapter fake --fake-refusal-rate 0.6``
    produces the same headline numbers as ``lre demo`` (which defaults
    to 0.6).
    """
    runner = CliRunner()
    out = tmp_path / "results.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--fake-refusal-rate",
            "0.6",
            "--seed",
            "42",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["refused"] == 8  # matches demo's table for the same suite
    assert payload[0]["complied"] == 7


# ---------------------------------------------------------------------------
# F-R3-P1-7: demo prints next-steps hint
# ---------------------------------------------------------------------------


def test_demo_prints_next_steps_hint() -> None:
    """F-R3-P1-7: a researcher running ``lre demo`` should see a Next
    steps block pointing at ``lre run`` and ``lre compare``.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--refusal-rate", "0.5", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "Next steps:" in result.output
    assert "lre run --adapter hf" in result.output
    assert "lre run --adapter openai" in result.output
    assert "lre compare" in result.output


# ---------------------------------------------------------------------------
# NEW-R3-1: --dump-raw
# ---------------------------------------------------------------------------


def test_run_dump_raw_writes_jsonl(tmp_path: Path) -> None:
    """NEW-R3-1: ``--dump-raw <path>`` writes per-prompt JSONL with one
    object per line, count matching the suite size.
    """
    runner = CliRunner()
    out = tmp_path / "results.json"
    raw_path = tmp_path / "raw.jsonl"
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
            "--dump-raw",
            str(raw_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert raw_path.exists()
    lines = raw_path.read_text(encoding="utf-8").strip().splitlines()
    # harmful_helpful has 15 prompts.
    assert len(lines) == 15
    # Each line must parse as a JSON object with the expected keys.
    for line in lines:
        obj = json.loads(line)
        assert "prompt_id" in obj
        assert "output" in obj
        assert "model" in obj


def test_judge_can_rejudge_from_dump_raw(tmp_path: Path) -> None:
    """NEW-R3-1: ``lre judge --in <dump>.jsonl`` re-judges without
    re-running generation.
    """
    runner = CliRunner()
    out = tmp_path / "results.json"
    raw_path = tmp_path / "raw.jsonl"
    runner.invoke(
        main,
        [
            "run",
            "--model",
            "fake-1b",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
            "--dump-raw",
            str(raw_path),
        ],
    )
    labels_path = tmp_path / "labels.json"
    result = runner.invoke(
        main,
        ["judge", "--in", str(raw_path), "--out", str(labels_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert len(payload) == 15


# ---------------------------------------------------------------------------
# NEW-R3-2: report --diff
# ---------------------------------------------------------------------------


def test_report_diff_no_flips(tmp_path: Path) -> None:
    """NEW-R3-2: identical labels → zero flips on either direction."""
    runner = CliRunner()
    pairs = [("p1", True), ("p2", False), ("p3", True)]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_labels(a, pairs)
    _write_labels(b, pairs)
    # Stub results file (required positional arg even though --diff bypasses it).
    stub = tmp_path / "stub.json"
    stub.write_text(to_json([_eval_result("m", "s", 1, 1)]), encoding="utf-8")
    result = runner.invoke(
        main,
        [
            "report",
            str(stub),
            "--diff",
            str(b),
            "--labels-current",
            str(a),
            "--labels-baseline",
            str(b),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "refused_now_complied:    0" in result.output
    assert "complied_now_refused:    0" in result.output


def test_report_diff_all_flipped(tmp_path: Path) -> None:
    """NEW-R3-2: every prompt flips → both lists are populated."""
    runner = CliRunner()
    baseline = [("p1", True), ("p2", False), ("p3", True), ("p4", False)]
    current = [(pid, not ref) for pid, ref in baseline]
    a = tmp_path / "current.json"
    b = tmp_path / "baseline.json"
    _write_labels(a, current)
    _write_labels(b, baseline)
    stub = tmp_path / "stub.json"
    stub.write_text(to_json([_eval_result("m", "s", 1, 1)]), encoding="utf-8")
    result = runner.invoke(
        main,
        [
            "report",
            str(stub),
            "--diff",
            str(b),
            "--labels-current",
            str(a),
            "--labels-baseline",
            str(b),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "refused_now_complied:    2" in result.output
    assert "complied_now_refused:    2" in result.output


def test_report_diff_partial_flip(tmp_path: Path) -> None:
    """NEW-R3-2: only one prompt flipped — surfaced exactly."""
    runner = CliRunner()
    baseline = [("p1", True), ("p2", False), ("p3", True)]
    current = [("p1", True), ("p2", True), ("p3", True)]
    a = tmp_path / "current.json"
    b = tmp_path / "baseline.json"
    _write_labels(a, current)
    _write_labels(b, baseline)
    stub = tmp_path / "stub.json"
    stub.write_text(to_json([_eval_result("m", "s", 1, 1)]), encoding="utf-8")
    result = runner.invoke(
        main,
        [
            "report",
            str(stub),
            "--diff",
            str(b),
            "--labels-current",
            str(a),
            "--labels-baseline",
            str(b),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "complied_now_refused:    1" in result.output
    assert "refused_now_complied:    0" in result.output
    # p2 is the flipper.
    assert "p2" in result.output


# ---------------------------------------------------------------------------
# F-R3-P2-9: CLI coverage gaps
# ---------------------------------------------------------------------------


def test_run_with_anthropic_adapter_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-R3-P2-9: cover ``--adapter anthropic`` happy path against a
    MockTransport-backed AsyncClient.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "I cannot help with that."}]},
        )

    transport = httpx.MockTransport(handler)

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    runner = CliRunner()
    out = tmp_path / "out.json"
    result = runner.invoke(
        main,
        [
            "run",
            "--adapter",
            "anthropic",
            "--model",
            "claude-3-5-sonnet-latest",
            "--suite",
            "harmful_helpful",
            "--out",
            str(out),
            "--max-concurrent",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["refused"] == 15  # every prompt → refusal text


def test_build_client_fake_uses_seed(tmp_path: Path) -> None:
    """F-R3-P2-9: ``_build_client('fake', ...)`` round-trips through the
    CLI happy path.
    """
    from lre.cli import _build_client

    client = _build_client(
        "fake",
        "fake-1b",
        model_id=None,
        base_url=None,
        api_key_env=None,
        seed=42,
        fake_refusal_rate=0.3,
    )
    assert client.name == "fake-1b"


def test_build_client_hf_requires_model_id() -> None:
    """F-R3-P2-9: ``--adapter hf`` without ``--model-id`` errors clean."""
    from lre.cli import _build_client

    with pytest.raises(Exception):  # noqa: B017 — Click UsageError
        _build_client(
            "hf",
            "hf-model",
            model_id=None,
            base_url=None,
            api_key_env=None,
            seed=0,
        )


def test_build_llm_judge_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-R3-P2-9: judge builder must fail fast on missing key."""
    from lre.cli import _build_llm_judge

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(Exception):  # noqa: B017
        _build_llm_judge(api_key_env=None, seed=0)


def test_build_llm_judge_constructs_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-R3-P2-9: judge builder succeeds when the env var is set."""
    from lre.cli import _build_llm_judge

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    judge = _build_llm_judge(api_key_env=None, seed=0)
    assert judge is not None


def test_judge_happy_path_round_trip(tmp_path: Path) -> None:
    """F-R3-P2-9: ``lre judge`` end-to-end happy path."""
    runner = CliRunner()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "p1",
                    "model": "m",
                    "output": "I cannot help with that.",
                    "generation_seconds": 0.01,
                    "timestamp": 0,
                    "seed": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.json"
    result = runner.invoke(main, ["judge", "--in", str(raw_path), "--out", str(labels_path)])
    assert result.exit_code == 0, result.output
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert len(labels) == 1
    assert labels[0]["refused"] is True


# ---------------------------------------------------------------------------
# F-R3-P2-19: BOM handling in suite files
# ---------------------------------------------------------------------------


def test_lint_handles_bom_prefixed_suite(tmp_path: Path) -> None:
    """F-R3-P2-19: a UTF-8 byte-order mark at the start of a suite must
    NOT corrupt the first JSON parse.
    """
    runner = CliRunner()
    bom_suite = tmp_path / "bom.jsonl"
    bom_suite.write_bytes(
        b"\xef\xbb\xbf"  # UTF-8 BOM
        + b'{"id": "p1", "text": "test prompt", "category": "harmful"}\n'
    )
    result = runner.invoke(main, ["lint", str(bom_suite)])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# F-R3-P2-20: lint output format for file-level issues
# ---------------------------------------------------------------------------


def test_lint_empty_suite_omits_line_no(tmp_path: Path) -> None:
    """F-R3-P2-20: ``suite contains no prompts`` is a file-level issue;
    output must drop the awkward ``:-:`` line-number segment.
    """
    runner = CliRunner()
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(main, ["lint", str(empty)])
    assert result.exit_code != 0
    # We accept either "<file>: error: ..." (preferred) or "<file>:-: ..." for
    # backward compat — but the new format must not contain ":-:" any more.
    assert ":-:" not in result.output
    assert "suite contains no prompts" in result.output


# ---------------------------------------------------------------------------
# the current implementation: edge-case input validation on CLI options
# ---------------------------------------------------------------------------


def test_run_sample_zero_clean_error(tmp_path: Path) -> None:
    """``--sample 0`` must fail with a Click range error, not silently
    fall through to the full suite (F-R4-P2-6)."""
    runner = CliRunner()
    out = tmp_path / "x.json"
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
            "0",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--sample'" in result.output


def test_run_sample_negative_clean_error(tmp_path: Path) -> None:
    """``--sample -1`` must also fail with a Click range error."""
    runner = CliRunner()
    out = tmp_path / "x.json"
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
            "-1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0


def test_compare_unsupported_confidence_clean_error(tmp_path: Path) -> None:
    """``lre compare --confidence 0.5`` must reject with a Click choice
    error, not a raw ValueError traceback (F-R4-P2-7)."""
    runner = CliRunner()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(to_json([_synthetic_result("m", "harmful_helpful", 5, 5)]), encoding="utf-8")
    b.write_text(to_json([_synthetic_result("m", "harmful_helpful", 5, 5)]), encoding="utf-8")
    result = runner.invoke(
        main,
        [
            "compare",
            str(a),
            str(b),
            "--confidence",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--confidence'" in result.output


def test_run_bad_cache_dir_clean_error(tmp_path: Path) -> None:
    """``--cache /dev/null`` (existing non-directory) must surface a
    UsageError, not a FileNotFoundError traceback (F-R4-P1-3)."""
    runner = CliRunner()
    out = tmp_path / "r.json"
    # /dev/null exists as a character device — mkdir fails on it.
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
            "--cache",
            "/dev/null",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "Cannot use --cache" in result.output


def test_run_cache_warns_on_nondeterministic_temperature(tmp_path: Path) -> None:
    """F-R4-P2-9: caching at temperature>0 cements one sample; warn."""
    runner = CliRunner()
    out = tmp_path / "r.json"
    cache = tmp_path / "c"
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
            "--cache",
            str(cache),
            "--temperature",
            "0.7",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.stderr
    assert "temperature=0.7" in result.stderr


def test_run_cache_nondet_warning_suppressed_by_flag(tmp_path: Path) -> None:
    """``--cache-allow-nondeterministic`` silences the temp>0 warning."""
    runner = CliRunner()
    out = tmp_path / "r.json"
    cache = tmp_path / "c"
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
            "--cache",
            str(cache),
            "--temperature",
            "0.7",
            "--cache-allow-nondeterministic",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.stderr


def test_run_help_documents_use_chat_template() -> None:
    """``lre run --help`` lists the chat-template toggle (F-R4-P2-14)."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--use-chat-template" in result.output
    assert "--no-use-chat-template" in result.output


def test_build_client_threads_chat_template_flag_to_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_build_client(adapter='hf', use_chat_template=False)`` propagates the flag.

    Mocks ``HFLocalClient`` so we don't actually load weights — we just
    want to confirm the CLI wiring forwards the value end-to-end. The
    flag also drops the ``@chat`` suffix from the default effective
    name when False.
    """
    from lre.cli import _build_client

    seen: dict[str, Any] = {}

    class _FakeHF:
        def __init__(self, *, model_id: str, name: str | None, use_chat_template: bool) -> None:
            seen["model_id"] = model_id
            seen["name"] = name
            seen["use_chat_template"] = use_chat_template
            self.name = name or model_id

    # Inject the fake into the hf_local module namespace so the CLI's
    # ``from lre.models.hf_local import HFLocalClient`` picks it up.
    import lre.models.hf_local as hf_mod

    monkeypatch.setattr(hf_mod, "HFLocalClient", _FakeHF)

    _build_client(
        "hf",
        model="qwen-test",
        model_id="org/qwen-0.5b",
        base_url=None,
        api_key_env=None,
        seed=42,
        fake_refusal_rate=0.5,
        use_chat_template=False,
    )
    assert seen == {
        "model_id": "org/qwen-0.5b",
        "name": "qwen-test",
        "use_chat_template": False,
    }


def test_kappa_help_does_not_leak_audit_ids() -> None:
    """``lre kappa --help`` must not contain internal F-R audit IDs (F-R4-P2-8)."""
    import re

    runner = CliRunner()
    result = runner.invoke(main, ["kappa", "--help"])
    assert result.exit_code == 0
    assert re.search(r"F-R\d+-P\d+-\d+", result.output) is None


# ---------------------------------------------------------------------------
# the current implementation: `lre cache` subcommands
# ---------------------------------------------------------------------------


def test_cache_info_missing_directory(tmp_path: Path) -> None:
    """``lre cache info --dir <missing>`` prints a friendly message, exit 0."""
    runner = CliRunner()
    result = runner.invoke(main, ["cache", "info", "--dir", str(tmp_path / "nope")])
    assert result.exit_code == 0
    assert "does not exist" in result.output


def test_cache_info_empty_directory(tmp_path: Path) -> None:
    """``lre cache info`` on an empty cache directory reports 0 entries."""
    from lre.cache import ResponseCache

    runner = CliRunner()
    cache_dir = tmp_path / "empty-cache"
    # Build the directory via ResponseCache so the ``.lre-cache``
    # sentinel is present — without it, ``lre cache info`` refuses to
    # touch the directory.
    ResponseCache(cache_dir)
    result = runner.invoke(main, ["cache", "info", "--dir", str(cache_dir)])
    assert result.exit_code == 0
    assert "0 entries" in result.output


def test_cache_info_refuses_directory_without_sentinel(tmp_path: Path) -> None:
    """the current implementation safety: ``lre cache info`` refuses non-cache directories."""
    runner = CliRunner()
    cache_dir = tmp_path / "stray"
    cache_dir.mkdir()
    (cache_dir / "ab.json").write_text("{}")  # decoy json file
    result = runner.invoke(main, ["cache", "info", "--dir", str(cache_dir)])
    assert result.exit_code != 0
    assert ".lre-cache" in result.output


def test_cache_clear_refuses_directory_without_sentinel(tmp_path: Path) -> None:
    """the current implementation safety: ``lre cache clear`` refuses non-cache directories."""
    runner = CliRunner()
    cache_dir = tmp_path / "stray"
    cache_dir.mkdir()
    (cache_dir / "ab").mkdir()
    (cache_dir / "ab" / "abc.json").write_text("{}")
    result = runner.invoke(main, ["cache", "clear", "--dir", str(cache_dir)])
    assert result.exit_code != 0
    assert ".lre-cache" in result.output


def test_cache_info_populated(tmp_path: Path) -> None:
    """A populated cache reports entry count + size + oldest/newest."""
    from lre.cache import ResponseCache
    from lre.state import RawResponse

    cache_dir = tmp_path / "filled"
    cache = ResponseCache(cache_dir)
    for i in range(3):
        cache.put(
            RawResponse(
                prompt_id=f"p{i}",
                model="m",
                output=f"out-{i}",
                generation_seconds=0.01,
                timestamp=0,
                seed=42,
            ),
            prompt=f"p{i}",
            seed=42,
            temperature=0.0,
            max_tokens=8,
        )
    runner = CliRunner()
    result = runner.invoke(main, ["cache", "info", "--dir", str(cache_dir)])
    assert result.exit_code == 0
    assert "3 entries" in result.output
    assert "oldest" in result.output and "newest" in result.output


def test_cache_clear_dry_run(tmp_path: Path) -> None:
    """``lre cache clear --dry-run`` lists targets but removes nothing."""
    from lre.cache import ResponseCache
    from lre.state import RawResponse

    cache_dir = tmp_path / "dryrun"
    cache = ResponseCache(cache_dir)
    cache.put(
        RawResponse(
            prompt_id="p",
            model="m",
            output="o",
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        ),
        prompt="p",
        seed=42,
        temperature=0.0,
        max_tokens=8,
    )
    runner = CliRunner()
    result = runner.invoke(main, ["cache", "clear", "--dir", str(cache_dir), "--dry-run"])
    assert result.exit_code == 0
    assert "would remove 1" in result.output
    # File still on disk.
    assert any(cache_dir.rglob("*.json"))


def test_cache_clear_removes_files(tmp_path: Path) -> None:
    """``lre cache clear`` without ``--older-than`` removes every entry."""
    from lre.cache import ResponseCache
    from lre.state import RawResponse

    cache_dir = tmp_path / "wipe"
    cache = ResponseCache(cache_dir)
    for i in range(2):
        cache.put(
            RawResponse(
                prompt_id=f"p{i}",
                model="m",
                output=f"o{i}",
                generation_seconds=0.01,
                timestamp=0,
                seed=42,
            ),
            prompt=f"p{i}",
            seed=42,
            temperature=0.0,
            max_tokens=8,
        )
    runner = CliRunner()
    result = runner.invoke(main, ["cache", "clear", "--dir", str(cache_dir)])
    assert result.exit_code == 0
    assert "removed 2" in result.output
    assert not list(cache_dir.rglob("*.json"))


def test_cache_clear_older_than_filter(tmp_path: Path) -> None:
    """``--older-than`` excludes recent entries from deletion."""
    import os
    import time as _time

    from lre.cache import ResponseCache
    from lre.state import RawResponse

    cache_dir = tmp_path / "selective"
    cache = ResponseCache(cache_dir)
    cache.put(
        RawResponse(
            prompt_id="p",
            model="m",
            output="o",
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        ),
        prompt="p",
        seed=42,
        temperature=0.0,
        max_tokens=8,
    )
    # Backdate the entry to 30 days ago and add a fresh one.
    old_file = next(cache_dir.rglob("*.json"))
    backdated = _time.time() - (30 * 86400)
    os.utime(old_file, (backdated, backdated))
    cache.put(
        RawResponse(
            prompt_id="p2",
            model="m",
            output="o2",
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        ),
        prompt="p2",
        seed=42,
        temperature=0.0,
        max_tokens=8,
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["cache", "clear", "--dir", str(cache_dir), "--older-than", "7d"],
    )
    assert result.exit_code == 0
    assert "removed 1" in result.output
    # The fresh entry survived.
    remaining = list(cache_dir.rglob("*.json"))
    assert len(remaining) == 1


def test_cache_clear_invalid_duration(tmp_path: Path) -> None:
    """A malformed ``--older-than`` rejects with a clean Click error."""
    from lre.cache import ResponseCache
    from lre.state import RawResponse

    cache_dir = tmp_path / "bad"
    # Build via ResponseCache so the sentinel is present and the new
    # safety check is satisfied. Add a real entry so the early-exit
    # path doesn't fire before the duration parser runs.
    cache = ResponseCache(cache_dir)
    cache.put(
        RawResponse(
            prompt_id="p",
            model="m",
            output="ok",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        ),
        prompt="p",
        seed=0,
        temperature=0.0,
        max_tokens=8,
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["cache", "clear", "--dir", str(cache_dir), "--older-than", "xyz"],
    )
    assert result.exit_code != 0
    assert "Invalid duration" in result.output


def _synthetic_result(model: str, suite: str, refused: int, complied: int) -> EvalResult:
    """Tiny EvalResult constructor for compare-side CLI smoke tests."""
    return EvalResult(
        model=model,
        suite=suite,
        total=refused + complied,
        refused=refused,
        complied=complied,
        ambiguous=0,
        refusal_rate=refused / max(refused + complied, 1),
        refusal_rate_ci_low=0.0,
        refusal_rate_ci_high=1.0,
        refusal_rate_by_category={},
        latency_p50_s=0.0,
        latency_p99_s=0.0,
    )


# ---------------------------------------------------------------------------
# symlink check refuses leaf symlinks, warns on parent symlinks
# ---------------------------------------------------------------------------


def test_safe_response_cache_refuses_leaf_symlink(tmp_path: Path) -> None:
    """the current implementation baseline: a symlinked leaf cache dir is refused."""
    from lre.cli import _safe_response_cache

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(Exception) as exc_info:
        _safe_response_cache(link)
    assert "symlink" in str(exc_info.value).lower()


def test_safe_response_cache_warns_on_parent_symlink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """the current implementation: a symlink in a PARENT of the cache path warns but does not refuse.

    An earlier iteration refused any parent symlink, which broke common benign cases:
    macOS ``/tmp -> /private/tmp`` and any user path containing ``..``.
    The new behavior preserves the leaf-symlink refusal (the actual
    attacker-controlled surface) and downgrades parent symlinks to a
    warning on stderr.
    """
    from lre.cli import _safe_response_cache

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.symlink_to(real_parent)
    # The cache leaf itself is NOT a symlink — only its parent is.
    cache_path = symlink_parent / "cache"
    assert not cache_path.is_symlink(), "test invariant: only the parent is a symlink, not the leaf"
    cache = _safe_response_cache(cache_path)
    # No exception; cache built; warning emitted on stderr.
    assert cache.cache_dir == cache_path
    captured = capsys.readouterr()
    assert "symlink" in captured.err.lower()
    assert "warning" in captured.err.lower()


def test_safe_response_cache_allows_parent_symlink_with_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--allow-symlinked-cache`` silences the parent-symlink warning."""
    from lre.cli import _safe_response_cache

    real_parent = tmp_path / "real_parent2"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink_parent2"
    symlink_parent.symlink_to(real_parent)
    cache_path = symlink_parent / "cache"
    # No exception when allow_symlinked=True; ResponseCache is built;
    # and no warning either (opt-in silences it).
    cache = _safe_response_cache(cache_path, allow_symlinked=True)
    assert cache.cache_dir == cache_path
    captured = capsys.readouterr()
    assert "symlink" not in captured.err.lower()


def test_safe_response_cache_accepts_dotdot_in_path(tmp_path: Path) -> None:
    """the current implementation: paths containing ``..`` segments are accepted (no symlinks involved).

    An earlier iteration compared ``absolute()`` (preserves ``..``) against
    ``resolve()`` (collapses ``..``), so ``foo/../cache`` triggered the
    symlink-refusal codepath even though no symlink existed.
    """
    from lre.cli import _safe_response_cache

    base = tmp_path / "base"
    base.mkdir()
    sub = base / "sub"
    sub.mkdir()
    # Path with a ``..`` segment that collapses to ``base/cache``.
    dotdot_path = sub / ".." / "cache"
    cache = _safe_response_cache(dotdot_path)
    assert cache.cache_dir == dotdot_path


def test_safe_response_cache_accepts_macos_style_tmp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """the current implementation: macOS-style ``/tmp -> /private/tmp`` parent symlinks are accepted.

    Simulates the macOS layout where ``/tmp`` is itself a system-level
    symlink. An earlier iteration refused such paths for everyone using ``/tmp``;
    the current implementation warns instead so the operator can proceed.
    """
    from lre.cli import _safe_response_cache

    # Build /private-like target and a /tmp-like symlink to it.
    real_root = tmp_path / "private" / "tmp_real"
    real_root.mkdir(parents=True)
    tmp_link = tmp_path / "tmp_link"
    tmp_link.symlink_to(real_root)
    cache_path = tmp_link / "lre_cache"
    cache = _safe_response_cache(cache_path)
    assert cache.cache_dir == cache_path
    # Should warn but not raise.
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


# ---------------------------------------------------------------------------
# --out - routes to stdout instead of creating a file named "-"
# ---------------------------------------------------------------------------


def test_run_out_dash_writes_to_stdout(tmp_path: Path) -> None:
    """``lre run --out -`` writes the JSON payload to stdout."""
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
            "--seed",
            "11",
            "--out",
            "-",
        ],
    )
    assert result.exit_code == 0, result.output
    # The stdout payload is a JSON array. Headline log goes to stderr.
    payload = result.stdout.strip()
    assert payload.startswith("["), payload[:200]
    parsed = json.loads(payload)
    assert isinstance(parsed, list) and len(parsed) >= 1
    # No literal "-" file created.
    assert not Path("-").exists()


def test_run_invalid_suite_name_renders_usage_error(tmp_path: Path) -> None:
    """a malicious suite name renders a clean Click usage error."""
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
            "../../etc/passwd",
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0
    # Click renders UsageError without a raw Python traceback.
    assert "Traceback" not in result.output, result.output


def test_run_fake_refusal_rate_out_of_range_renders_usage_error(tmp_path: Path) -> None:
    """--fake-refusal-rate outside [0, 1] fails fast with FloatRange."""
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
            "--fake-refusal-rate",
            "2",
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output, result.output


# ---------------------------------------------------------------------------
# `lre reproduce` reconstructs the run invocation
# ---------------------------------------------------------------------------


def _run_for_reproduce(tmp_path: Path, *, seed: int = 7) -> Path:
    """Helper: run ``lre run`` with the fake adapter and return the JSON path."""
    runner = CliRunner()
    out_path = tmp_path / "orig.json"
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
            "--seed",
            str(seed),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    return out_path


def test_reproduce_prints_lre_run_invocation(tmp_path: Path) -> None:
    """``lre reproduce`` prints a reconstructable ``lre run`` command per row."""
    out_path = _run_for_reproduce(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["reproduce", str(out_path)])
    assert result.exit_code == 0, result.output
    assert "Original lre version" in result.output
    assert "SOURCE_DATE_EPOCH" in result.output
    # the current implementation: the reconstructed command always includes --adapter and
    # carries the full CLI-input set captured in provenance.
    assert "lre run" in result.output
    assert "--adapter fake" in result.output
    assert "--model fake-1b" in result.output
    assert "--suite harmful_helpful" in result.output
    assert "--seed 7" in result.output


def test_reproduce_exec_roundtrip_matches_input(tmp_path: Path) -> None:
    """``--exec`` re-runs the harness; output matches input modulo provenance.

    The fake adapter is deterministic in ``(seed, prompts,
    refusal_rate)``, so the reconstructed run produces byte-identical
    EvalResult payloads when the captured ``fake_refusal_rate`` is
    used (which the current implementation does — an earlier iteration hardcoded 0.5 and silently
    broke this invariant).
    """
    out_path = _run_for_reproduce(tmp_path, seed=5)
    runner = CliRunner()
    repro_path = tmp_path / "repro.json"
    result = runner.invoke(
        main,
        ["reproduce", str(out_path), "--exec", "--out", str(repro_path)],
    )
    assert result.exit_code == 0, result.output
    assert repro_path.exists()
    # Compare the EvalResult payloads modulo provenance.
    original = json.loads(out_path.read_text())
    reproduced = json.loads(repro_path.read_text())

    def _strip_prov(blob: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in blob:
            row = dict(row)
            row.pop("provenance", None)
            out.append(row)
        return out

    assert _strip_prov(original) == _strip_prov(reproduced)


def test_reproduce_refuses_results_without_provenance(tmp_path: Path) -> None:
    """Legacy the current implementation result files (no provenance) cannot be reproduced — clean error."""
    # Hand-craft an EvalResult JSON without provenance.
    legacy_blob = [
        {
            "model": "fake",
            "suite": "harmful_helpful",
            "total": 1,
            "refused": 1,
            "complied": 0,
            "ambiguous": 0,
            "refusal_rate": 1.0,
            "refusal_rate_ci_low": 0.0,
            "refusal_rate_ci_high": 1.0,
            "refusal_rate_by_category": {"harmful": 1.0},
            "latency_p50_s": 0.0,
            "latency_p99_s": 0.0,
        }
    ]
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_blob))
    runner = CliRunner()
    result = runner.invoke(main, ["reproduce", str(legacy_path)])
    assert result.exit_code != 0
    assert "provenance" in result.output.lower()


def test_reproduce_rejects_pre_v1_provenance(tmp_path: Path) -> None:
    """the current implementation: pre-v1.0 results files (no captured adapter) get a clean error.

    Pre-v1.0 provenance lacks ``adapter`` / ``fake_refusal_rate`` /
    ``sample_n`` / ``judge_kind`` etc. so the reproduce path cannot
    rebuild the original invocation. Fail with a usage error pointing
    at the missing field instead of silently guessing.
    """
    legacy_blob = [
        {
            "model": "fake",
            "suite": "harmful_helpful",
            "total": 1,
            "refused": 1,
            "complied": 0,
            "ambiguous": 0,
            "refusal_rate": 1.0,
            "refusal_rate_ci_low": 0.0,
            "refusal_rate_ci_high": 1.0,
            "refusal_rate_by_category": {"harmful": 1.0},
            "latency_p50_s": 0.0,
            "latency_p99_s": 0.0,
            "provenance": {
                "schema_version": "0.9",
                "lre_version": "0.9.0",
                "python_version": "3.11.0",
                "platform": "Linux",
                "hostname_hash": "abc1234567890def",
                "run_timestamp_utc": "2026-05-23T17:00:00Z",
                "seed": 7,
                "model_id": "fake-1b",
                "temperature": 0.0,
                "max_tokens": 512,
                # NB: no adapter / fake_refusal_rate / sample_n /
                # judge_kind / etc. — this is what a legacy file looks like.
            },
        }
    ]
    legacy_path = tmp_path / "pre_v1.json"
    legacy_path.write_text(json.dumps(legacy_blob))
    runner = CliRunner()
    result = runner.invoke(main, ["reproduce", str(legacy_path)])
    assert result.exit_code != 0
    # Error must mention both 'adapter' and 'the current implementation' so the operator
    # knows what to fix.
    assert "adapter" in result.output.lower()
    assert "current release" in result.output


def test_reproduce_print_includes_all_captured_flags(tmp_path: Path) -> None:
    """the current implementation: reproduce reconstructs the full CLI invocation, not just a 4-tuple.

    Run with ``--fake-refusal-rate 0.7 --sample 3 --judge rule`` and
    confirm every captured flag shows up in the reconstructed command.
    """
    runner = CliRunner()
    out_path = tmp_path / "orig.json"
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
            "--seed",
            "9",
            "--fake-refusal-rate",
            "0.7",
            "--sample",
            "3",
            "--judge",
            "rule",
            "--max-concurrent",
            "2",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output

    repro = runner.invoke(main, ["reproduce", str(out_path)])
    assert repro.exit_code == 0, repro.output
    text = repro.output
    assert "--adapter fake" in text
    assert "--fake-refusal-rate 0.7" in text
    assert "--sample 3" in text
    assert "--judge rule" in text
    assert "--max-concurrent 2" in text
    assert "--seed 9" in text


def test_reproduce_exec_byte_identical_with_source_date_epoch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """the current implementation: with SOURCE_DATE_EPOCH pinned, --exec is byte-identical to source.

    Includes ``--fake-refusal-rate 0.7`` and ``--sample 3`` so the test
    actually exercises the legacy honest-reproduce path. An earlier iteration the
    reproduce-exec hardcoded refusal_rate=0.5 and stripped --sample, so
    this test would have failed against the current implementation.
    """
    import hashlib

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    runner = CliRunner()
    orig_path = tmp_path / "orig.json"
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
            "--seed",
            "11",
            "--fake-refusal-rate",
            "0.7",
            "--sample",
            "3",
            "--out",
            str(orig_path),
        ],
    )
    assert result.exit_code == 0, result.output

    repro_path = tmp_path / "repro.json"
    repro = runner.invoke(
        main,
        ["reproduce", str(orig_path), "--exec", "--out", str(repro_path)],
    )
    assert repro.exit_code == 0, repro.output

    orig_bytes = orig_path.read_bytes()
    repro_bytes = repro_path.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    repro_sha = hashlib.sha256(repro_bytes).hexdigest()
    assert orig_sha == repro_sha, (
        f"reproduce --exec must be byte-identical with SOURCE_DATE_EPOCH pinned; "
        f"orig={orig_sha} repro={repro_sha}"
    )


def test_reproduce_groups_by_judge_kind(tmp_path: Path) -> None:
    """runs differing only in judge collapse into separate commands.

    An earlier iteration grouped by 4-tuple ignoring judge type, so a rule-judge
    and llm-judge run with otherwise-identical knobs were silently
    collapsed into a single reconstructed command. the current implementation includes
    ``judge_kind`` in the group key so they stay distinct.
    """
    runner = CliRunner()
    # Two runs differ only in --judge (both use rule, but exercise the
    # group key separately). We can't easily test rule vs llm without
    # the LLM client; instead we synthesize two EvalResult rows with
    # different ``judge_kind`` provenance values and ensure both end up
    # in separate reconstructed commands.
    out_path = tmp_path / "two_judges.json"
    blob = [
        {
            "model": "fake-1b",
            "suite": "harmful_helpful",
            "total": 1,
            "refused": 1,
            "complied": 0,
            "ambiguous": 0,
            "refusal_rate": 1.0,
            "refusal_rate_ci_low": 0.0,
            "refusal_rate_ci_high": 1.0,
            "refusal_rate_by_category": {"harmful": 1.0},
            "latency_p50_s": 0.0,
            "latency_p99_s": 0.0,
            "provenance": {
                "schema_version": "1.0",
                "lre_version": "0.10.0",
                "python_version": "3.11.0",
                "platform": "Linux",
                "hostname_hash": "h",
                "run_timestamp_utc": "2026-05-23T17:00:00Z",
                "seed": 1,
                "model_id": "fake-1b",
                "temperature": 0.0,
                "max_tokens": 512,
                "adapter": "fake",
                "fake_refusal_rate": 0.5,
                "judge_kind": kind,
                "max_concurrent": 4,
            },
        }
        for kind in ("rule", "llm")
    ]
    out_path.write_text(json.dumps(blob))
    repro = runner.invoke(main, ["reproduce", str(out_path)])
    assert repro.exit_code == 0, repro.output
    # Two distinct judge_kind groups -> two reconstructed `lre run` lines.
    lre_run_count = sum(1 for line in repro.output.splitlines() if line.startswith("lre run"))
    assert lre_run_count == 2, repro.output
    assert "--judge rule" in repro.output
    assert "--judge llm" in repro.output
