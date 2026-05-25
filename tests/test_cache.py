"""Tests for :mod:`lre.cache`.

The cache is the lowest-friction feature shipped in the current implementation: a user with
a ``--cache .lre-cache/`` flag spends API quota once and replays the
same generation locally on every subsequent run. These tests pin the
key derivation, the on-disk layout, and the runner's wiring.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from lre.cache import ResponseCache, _cache_key
from lre.runner import run_eval
from lre.state import Prompt, RawResponse, RunConfig


class _RecordingClient:
    """Test double — records every ``generate`` call so we can assert call counts."""

    name = "rec-1b"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> RawResponse:
        self.calls += 1
        return RawResponse(
            prompt_id="",
            model=self.name,
            output=f"out-{prompt}-{seed}",
            generation_seconds=0.01,
            timestamp=0,
            seed=seed,
        )


class _AssertNoCallClient:
    """Test double that fails the test if ``generate`` is ever called."""

    name = "rec-1b"  # same name so the cache key matches a recorded entry

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> RawResponse:  # pragma: no cover - failure path
        raise AssertionError("Cache miss — generate() must not be called on a hit")


def test_cache_key_includes_all_relevant_fields() -> None:
    """Different inputs must produce different keys."""
    base = _cache_key(model="m", prompt="hello", seed=0, temperature=0.0, max_tokens=10)
    different_model = _cache_key(model="x", prompt="hello", seed=0, temperature=0.0, max_tokens=10)
    different_prompt = _cache_key(model="m", prompt="HI", seed=0, temperature=0.0, max_tokens=10)
    different_seed = _cache_key(model="m", prompt="hello", seed=1, temperature=0.0, max_tokens=10)
    different_temp = _cache_key(model="m", prompt="hello", seed=0, temperature=0.5, max_tokens=10)
    different_tok = _cache_key(model="m", prompt="hello", seed=0, temperature=0.0, max_tokens=11)
    keys = {base, different_model, different_prompt, different_seed, different_temp, different_tok}
    assert len(keys) == 6


def test_cache_key_distinguishes_chat_template_flag() -> None:
    """F-R4-P1-2: an effective-name that bakes in the chat-template suffix
    must produce a different cache key from the bare model id.

    Without this, an ``HFLocalClient`` with ``use_chat_template=True``
    and one with ``use_chat_template=False`` would silently share cache
    entries despite producing different completions from different
    prompts. The contract is: distinct effective names => distinct keys.
    """
    bare = _cache_key(model="org/qwen-0.5b", prompt="hi", seed=0, temperature=0.0, max_tokens=8)
    chat = _cache_key(
        model="org/qwen-0.5b@chat", prompt="hi", seed=0, temperature=0.0, max_tokens=8
    )
    assert bare != chat


def test_cache_key_extra_parts_distinguish_keys() -> None:
    """The ``extra_key_parts`` tuple participates in the hash.

    Caller-supplied discriminators (e.g. a tokenizer revision, a
    serving-side system prompt) widen the key without forcing every
    call site through the ``name@suffix`` convention.
    """
    base = _cache_key(model="m", prompt="hi", seed=0, temperature=0.0, max_tokens=8)
    widened = _cache_key(
        model="m",
        prompt="hi",
        seed=0,
        temperature=0.0,
        max_tokens=8,
        extra_key_parts=("rev=v2",),
    )
    assert base != widened
    # Default empty tuple matches the legacy key format byte-for-byte,
    # so the current implementation cache directories remain readable.
    legacy = _cache_key(
        model="m",
        prompt="hi",
        seed=0,
        temperature=0.0,
        max_tokens=8,
        extra_key_parts=(),
    )
    assert base == legacy


def test_cache_atomic_write_leaves_no_partial_files(tmp_path: Path) -> None:
    """F-R4-P2-10: a .tmp file left from a killed write must not be served as a hit.

    Simulates ``put`` crashing partway by writing a ``<key>.json.tmp``
    by hand and never renaming it. The cache should still report a miss
    on ``get`` (the .tmp is not the canonical name) and not return a
    corrupt entry.
    """
    cache = ResponseCache(tmp_path / "atomic")
    key = _cache_key(model="m", prompt="p", seed=0, temperature=0.0, max_tokens=8)
    shard = tmp_path / "atomic" / key[:2]
    shard.mkdir(parents=True, exist_ok=True)
    tmp_file = shard / f"{key}.json.tmp"
    tmp_file.write_text("{ partial data, write interrupted")
    # No <key>.json — only the staged .tmp exists.
    assert cache.get("m", "p", 0, 0.0, 8) is None
    assert cache.stats()["misses"] == 1
    # A subsequent successful put completes atomically.
    rr = RawResponse(
        prompt_id="p",
        model="m",
        output="ok",
        generation_seconds=0.01,
        timestamp=0,
        seed=0,
    )
    cache.put(rr, prompt="p", seed=0, temperature=0.0, max_tokens=8)
    final = shard / f"{key}.json"
    assert final.is_file()
    # Re-read works — atomic rename produced a valid JSON.
    fetched = cache.get("m", "p", 0, 0.0, 8)
    assert fetched is not None
    assert fetched.output == "ok"


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c")
    rr = RawResponse(
        prompt_id="p1",
        model="m",
        output="hi",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    assert cache.get("m", "hello", 42, 0.0, 16) is None
    cache.put(rr, prompt="hello", seed=42, temperature=0.0, max_tokens=16)
    fetched = cache.get("m", "hello", 42, 0.0, 16)
    assert fetched is not None
    assert fetched.output == "hi"
    stats = cache.stats()
    assert stats == {"hits": 1, "misses": 1, "writes": 1}


def test_cache_storage_layout_is_sharded(tmp_path: Path) -> None:
    """Files are stored under ``<cache_dir>/<key[:2]>/<key>.json``."""
    cache = ResponseCache(tmp_path / "shards")
    rr = RawResponse(
        prompt_id="p1",
        model="m",
        output="hi",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    cache.put(rr, prompt="hello", seed=42, temperature=0.0, max_tokens=16)
    key = _cache_key(model="m", prompt="hello", seed=42, temperature=0.0, max_tokens=16)
    expected = tmp_path / "shards" / key[:2] / f"{key}.json"
    assert expected.is_file()
    payload = json.loads(expected.read_text())
    assert payload["model"] == "m"
    assert payload["output"] == "hi"


def test_cache_persists_across_runs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "persistent"
    a = ResponseCache(cache_dir)
    rr = RawResponse(
        prompt_id="p1",
        model="m",
        output="echo",
        generation_seconds=0.01,
        timestamp=0,
        seed=42,
    )
    a.put(rr, prompt="prompt", seed=42, temperature=0.1, max_tokens=64)
    del a
    b = ResponseCache(cache_dir)
    fetched = b.get("m", "prompt", 42, 0.1, 64)
    assert fetched is not None
    assert fetched.output == "echo"


def test_cache_corrupt_file_is_treated_as_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "corrupt")
    key = _cache_key(model="m", prompt="p", seed=0, temperature=0.0, max_tokens=16)
    shard = tmp_path / "corrupt" / key[:2]
    shard.mkdir(parents=True, exist_ok=True)
    (shard / f"{key}.json").write_text("{not valid json")
    assert cache.get("m", "p", 0, 0.0, 16) is None


def test_cache_hit_skips_client_call(tmp_path: Path) -> None:
    """A run with a populated cache must not invoke the model client."""
    cache = ResponseCache(tmp_path / "skip")
    rr = RawResponse(
        prompt_id="",
        model="rec-1b",
        output="cached!",
        generation_seconds=0.01,
        timestamp=0,
        seed=0,
    )
    prompt_text = "synthetic"
    cache.put(rr, prompt=prompt_text, seed=42, temperature=0.0, max_tokens=64)

    client = _AssertNoCallClient()
    prompt = Prompt(id="p1", suite="demo", text=prompt_text, category="helpful")
    config = RunConfig(model="rec-1b", suites=["demo"], max_tokens=64, seed=42)

    responses = asyncio.run(run_eval(client, "demo", config, cache=cache, prompts=[prompt]))
    assert len(responses) == 1
    assert responses[0].output == "cached!"
    assert responses[0].generation_seconds == 0.0  # zero-latency replay marker
    assert cache.stats()["hits"] == 1


def test_cache_writes_after_real_generation(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "writes")
    client = _RecordingClient()
    prompt = Prompt(id="p1", suite="demo", text="hello", category="helpful")
    config = RunConfig(model="rec-1b", suites=["demo"], max_tokens=64)

    # First run — cold cache.
    responses_a = asyncio.run(run_eval(client, "demo", config, cache=cache, prompts=[prompt]))
    assert client.calls == 1
    assert responses_a[0].output.startswith("out-hello-")
    assert cache.stats()["misses"] == 1
    assert cache.stats()["writes"] == 1

    # Second run — should hit the cache, NOT call client again.
    responses_b = asyncio.run(run_eval(client, "demo", config, cache=cache, prompts=[prompt]))
    assert client.calls == 1  # unchanged
    assert responses_b[0].output == responses_a[0].output
    assert cache.stats()["hits"] == 1


def test_cache_dir_is_created_on_demand(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "cache"
    assert not nested.exists()
    cache = ResponseCache(nested)
    assert nested.is_dir()
    # And it works:
    rr = RawResponse(
        prompt_id="p",
        model="m",
        output="ok",
        generation_seconds=0.01,
        timestamp=0,
        seed=0,
    )
    cache.put(rr, prompt="x", seed=0, temperature=0.0, max_tokens=16)
    assert cache.get("m", "x", 0, 0.0, 16) is not None


def test_cache_error_sentinels_are_not_cached(tmp_path: Path) -> None:
    """Generation errors yield ``generation_seconds=-1`` sentinels; do not cache."""
    cache = ResponseCache(tmp_path / "no-errors")

    class _ErrorClient:
        name = "err"

        async def generate(self, *args: Any, **kwargs: Any) -> RawResponse:
            raise RuntimeError("model is sad")

    prompt = Prompt(id="p1", suite="demo", text="hi", category="helpful")
    config = RunConfig(model="err", suites=["demo"], max_tokens=8)
    responses = asyncio.run(run_eval(_ErrorClient(), "demo", config, cache=cache, prompts=[prompt]))
    assert responses[0].generation_seconds == -1.0
    # Nothing was written to the cache (writes == 0):
    assert cache.stats()["writes"] == 0


def test_cli_run_with_cache_logs_stats(tmp_path: Path) -> None:
    """``lre run --cache <dir>`` should report cache stats on stdout."""
    from click.testing import CliRunner

    from lre.cli import main

    cache_dir = tmp_path / "cli-cache"
    out = tmp_path / "r.json"
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
            "--cache",
            str(cache_dir),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cache:" in result.output
    assert "writes" in result.output
    # Second run — every prompt should be a hit.
    out2 = tmp_path / "r2.json"
    result2 = runner.invoke(
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
            str(cache_dir),
            "--out",
            str(out2),
        ],
    )
    assert result2.exit_code == 0, result2.output
    # Should be hits in the second run.
    assert "Cache:" in result2.output
    assert " 0 misses" in result2.output or " 0 writes" in result2.output


# pytest-friendly module-level sanity import guard so collection fails
# loudly if any of the new modules vanish under us.
def test_module_imports() -> None:
    import lre.cache
    import lre.defense_in_depth
    import lre.provenance  # noqa: F401


# ---------------------------------------------------------------------------
# the current implementation contract: canonical-JSON key + sentinel + collision safety
# ---------------------------------------------------------------------------


def test_cache_key_resists_delimiter_collision() -> None:
    """the current implementation: pathological ``|``-laden inputs must not collide.

    Under the an earlier iteration pipe-joined key, ``model='a|b' + prompt='c'`` and
    ``model='a' + prompt='b|c'`` hashed to the same bytes. The
    canonical-JSON key keeps them distinct.
    """
    collide_a = _cache_key(model="a|b", prompt="c", seed=0, temperature=0.0, max_tokens=8)
    collide_b = _cache_key(model="a", prompt="b|c", seed=0, temperature=0.0, max_tokens=8)
    assert collide_a != collide_b


def test_cache_sentinel_is_written_on_construction(tmp_path: Path) -> None:
    """the current implementation: ``ResponseCache`` drops a ``.lre-cache`` sentinel."""
    from lre.cache import SENTINEL_FILENAME, is_lre_cache_dir

    cache_dir = tmp_path / "with-sentinel"
    ResponseCache(cache_dir)
    assert (cache_dir / SENTINEL_FILENAME).is_file()
    assert is_lre_cache_dir(cache_dir)


def test_cache_sentinel_is_absent_on_arbitrary_directory(tmp_path: Path) -> None:
    """Directories the harness did not create must not satisfy the sentinel check."""
    from lre.cache import is_lre_cache_dir

    cache_dir = tmp_path / "plain"
    cache_dir.mkdir()
    (cache_dir / "ab.json").write_text("{}")
    assert not is_lre_cache_dir(cache_dir)


def test_cache_sentinel_rejects_invalid_content(tmp_path: Path) -> None:
    """a bare-touched ``.lre-cache`` file no longer passes.

    An earlier iteration ``is_lre_cache_dir`` only checked existence. An attacker
    (or an accidental ``touch .lre-cache``) could pass the check
    without a real sentinel.
    """
    from lre.cache import SENTINEL_FILENAME, is_lre_cache_dir

    cache_dir = tmp_path / "fake-sentinel"
    cache_dir.mkdir()
    sentinel = cache_dir / SENTINEL_FILENAME
    # Empty file — no JSON.
    sentinel.write_text("")
    assert not is_lre_cache_dir(cache_dir)
    # Valid JSON but wrong purpose.
    sentinel.write_text('{"purpose": "something else"}')
    assert not is_lre_cache_dir(cache_dir)
    # Valid JSON, correct purpose, but no schema_version.
    sentinel.write_text('{"purpose": "lm-refusal-eval response cache"}')
    assert not is_lre_cache_dir(cache_dir)
    # Real sentinel content — passes.
    sentinel.write_text('{"schema_version": "1", "purpose": "lm-refusal-eval response cache"}')
    assert is_lre_cache_dir(cache_dir)


def test_cli_cache_migrate_writes_sentinel(tmp_path: Path) -> None:
    """``lre cache migrate`` writes the sentinel on a
    an earlier iteration cache directory that has cached entries but no marker.
    """
    from click.testing import CliRunner

    from lre.cache import SENTINEL_FILENAME, is_lre_cache_dir
    from lre.cli import main

    cache_dir = tmp_path / "pre-v07"
    cache_dir.mkdir()
    (cache_dir / "ab").mkdir()
    (cache_dir / "ab" / "abcd.json").write_text(
        '{"prompt_id": "p", "model": "m", "output": "ok", '
        '"generation_seconds": 0.01, "timestamp": 0, "seed": 0}'
    )
    assert not is_lre_cache_dir(cache_dir)
    runner = CliRunner()
    result = runner.invoke(main, ["cache", "migrate", "--dir", str(cache_dir)])
    assert result.exit_code == 0, result.output
    assert (cache_dir / SENTINEL_FILENAME).is_file()
    assert is_lre_cache_dir(cache_dir)


def test_cli_cache_migrate_purge_stale_removes_invalid_entries(tmp_path: Path) -> None:
    """``--purge-stale`` deletes entries that don't parse as RawResponse."""
    from click.testing import CliRunner

    from lre.cli import main

    cache_dir = tmp_path / "purge"
    cache_dir.mkdir()
    (cache_dir / "ab").mkdir()
    # Valid row.
    (cache_dir / "ab" / "ok.json").write_text(
        '{"prompt_id": "p", "model": "m", "output": "ok", '
        '"generation_seconds": 0.01, "timestamp": 0, "seed": 0}'
    )
    # Stale: missing required ``timestamp`` and ``seed``.
    (cache_dir / "ab" / "stale.json").write_text(
        '{"prompt_id": "p", "model": "m", "output": "stale"}'
    )
    runner = CliRunner()
    result = runner.invoke(main, ["cache", "migrate", "--dir", str(cache_dir), "--purge-stale"])
    assert result.exit_code == 0, result.output
    assert (cache_dir / "ab" / "ok.json").is_file()
    assert not (cache_dir / "ab" / "stale.json").is_file()
    assert "purged 1" in result.output


def test_cache_concurrent_writers_do_not_clobber(tmp_path: Path) -> None:
    """Two concurrent ``put`` calls on the same key must both succeed."""

    async def _writer(cache: ResponseCache, output: str) -> None:
        rr = RawResponse(
            prompt_id="p1",
            model="m",
            output=output,
            generation_seconds=0.01,
            timestamp=0,
            seed=42,
        )
        cache.put(rr, prompt="hello", seed=42, temperature=0.0, max_tokens=16)

    async def _main() -> None:
        cache = ResponseCache(tmp_path / "concurrent")
        await asyncio.gather(*(_writer(cache, f"out-{i}") for i in range(8)))
        fetched = cache.get("m", "hello", 42, 0.0, 16)
        assert fetched is not None
        assert fetched.output.startswith("out-")

    asyncio.run(_main())
