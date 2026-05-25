"""Content-addressed cache for :class:`~lre.state.RawResponse` objects.

A working researcher iterating on judge prompts or scaling-laws plots
re-runs the same ``(prompt, model, seed, temperature, max_tokens)`` tuple
many times. Without a cache, every iteration burns API quota and time.

Design
------
* The cache key is ``SHA256(canonical_json({"model", "prompt", "seed",
  "temperature", "max_tokens", "extras"}))`` — the canonical JSON is
  built with ``json.dumps(..., sort_keys=True, ensure_ascii=False)`` so
  no separator character (``|``, ``\\n``, etc.) can be smuggled inside
  a field to silently collide two distinct inputs. The previous v0.6
  format ``f"{model}|{prompt}|..."`` was ambiguous: ``model='a|b' +
  prompt='c'`` and ``model='a' + prompt='b|c'`` produced the same key.
  Canonical JSON closes that hole. ``extras`` is a tuple of arbitrary
  additional discriminators provided by the caller (see
  ``extra_key_parts``); the runner threads it through so adapters can
  disambiguate options that change observable behaviour but are not
  already in the canonical 5-tuple — e.g. ``HFLocalClient.use_chat_template``
  flips the prompt envelope without changing the ``prompt`` argument
  itself, so omitting it from the key would silently serve a wrong
  completion. The recommended convention is to bake the discriminator
  into ``client.name`` (e.g. ``"Qwen2-0.5B-Instruct@chat"``) so the
  effective name participates in the existing ``model`` slot; callers
  with one-off requirements may pass ``extra_key_parts`` instead.
* Storage layout is ``<cache_dir>/<key[:2]>/<key>.json`` — sharding by
  the first two hex chars keeps any one directory manageable even with
  millions of entries on disk.
* The on-disk format is the JSON-mode :meth:`pydantic.BaseModel.model_dump`
  of the :class:`RawResponse`, written with ``sort_keys=True`` so cached
  files are byte-stable. Writes are atomic AND collision-safe under
  concurrent writers: ``put`` stages to a unique ``tempfile.mkstemp``
  filename inside the shard directory, ``os.fsync``s the staged file,
  then ``os.replace``s into the canonical name and best-effort fsyncs
  the parent directory. No two concurrent writers ever race on the
  same temp filename. No proprietary marshalling, no pickle — anyone
  can ``cat`` an entry and read it.
* :class:`ResponseCache` writes a ``.lre-cache`` sentinel file on
  construction. The ``lre cache info`` and ``lre cache clear`` CLI
  commands refuse to operate on directories without that sentinel —
  a safety net so ``lre cache clear --dir ~/Documents`` cannot
  accidentally delete arbitrary ``*.json`` files.
* :class:`ResponseCache` is process-local; concurrent writers from
  different processes are safe (each writes its own shard file) but
  ordering is undefined. The cache is best-effort, not a database.
* :meth:`ResponseCache.stats` tracks ``hits / misses / writes`` so the
  CLI can print a summary after a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from lre.state import RawResponse

logger = logging.getLogger(__name__)

# Filename of the sentinel file dropped at the cache root. Its presence
# is the precondition for ``lre cache clear`` / ``lre cache info`` to
# touch the directory.
SENTINEL_FILENAME = ".lre-cache"

# Body of the sentinel file — JSON so we can evolve the schema later
# without renaming the file.
_SENTINEL_BODY = '{"schema_version": "1", "purpose": "lm-refusal-eval response cache"}\n'


def _cache_key(
    *,
    model: str,
    prompt: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    extra_key_parts: tuple[str, ...] = (),
) -> str:
    """Return the SHA-256 hex digest of the canonical-JSON key.

    The canonical form is a JSON object with sorted keys, no ASCII
    escaping, and ``allow_nan=False``. ``temperature`` is rounded to
    six decimals so two callers that pass ``0.0`` and ``0.000001`` do
    not silently share a cache entry. The list-valued ``extras`` field
    is omitted when empty so the v0.6 empty-tuple default still
    produces a stable key (its hash differs from the v0.6 pipe-joined
    digest — caches written by v0.6 will be re-populated on the next
    run, which is the intended migration).
    """
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "seed": seed,
        "temperature": round(temperature, 6),
        "max_tokens": max_tokens,
    }
    if extra_key_parts:
        payload["extras"] = list(extra_key_parts)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_lre_cache_dir(cache_dir: Path) -> bool:
    """Return True if ``cache_dir`` carries a valid ``.lre-cache`` sentinel.

    The CLI uses this to confirm a directory was created by
    :class:`ResponseCache` before running ``cache clear`` / ``cache
    info`` on it — protects against ``--dir ~/Documents`` accidents.

    v0.8 (P1-12): the sentinel content is also validated. Pre-v0.8
    only the file's presence was checked, so any 0-byte ``.lre-cache``
    file would pass — including a touched-by-hand sentinel created
    BEFORE the cache wrote any real data. Now the sentinel must parse
    as JSON with the expected ``purpose`` and ``schema_version`` keys.
    """
    sentinel = Path(cache_dir) / SENTINEL_FILENAME
    if not sentinel.is_file():
        return False
    try:
        body = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(body, dict):
        return False
    return body.get("purpose") == "lm-refusal-eval response cache" and isinstance(
        body.get("schema_version"), str
    )


class ResponseCache:
    """Content-addressed cache for :class:`RawResponse` objects.

    Parameters
    ----------
    cache_dir:
        Root directory for cached files. Created on demand if absent.
        A ``.lre-cache`` sentinel file is written on construction so the
        CLI can later confirm the directory is genuinely an ``lre``
        cache before running destructive ``cache clear`` operations.

    Example
    -------
    >>> from pathlib import Path
    >>> cache = ResponseCache(Path("/tmp/lre-cache"))
    >>> cached = cache.get("fake-1b", "hello", 42, 0.0, 256)
    >>> if cached is None:
    ...     ...  # call the real model
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        # Surface OSError to the caller rather than crashing on
        # construction — the CLI wraps this in ``click.UsageError`` so a
        # bad ``--cache`` path renders as a one-line message instead of
        # a traceback.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_sentinel()
        self._hits = 0
        self._misses = 0
        self._writes = 0

    # ------------------------------------------------------------------
    # Internal layout
    # ------------------------------------------------------------------

    def _write_sentinel(self) -> None:
        """Drop the ``.lre-cache`` marker atomically on first use.

        Idempotent: if the sentinel already exists, the file is left
        alone (its presence is what matters, not its mtime).
        """
        sentinel = self.cache_dir / SENTINEL_FILENAME
        if sentinel.is_file():
            return
        # Atomic write via tempfile + os.replace so a concurrent
        # construction does not produce a half-written sentinel.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.cache_dir), prefix=".lre-cache.", suffix=".tmp"
        )
        try:
            os.write(fd, _SENTINEL_BODY.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, sentinel)

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        model: str,
        prompt: str,
        seed: int,
        temperature: float,
        max_tokens: int,
        extra_key_parts: tuple[str, ...] = (),
    ) -> RawResponse | None:
        """Return a cached :class:`RawResponse`, or ``None`` on a miss.

        Corrupt cache files (truncated JSON, schema drift from an older
        version) count as misses rather than raising — the cache is
        best-effort. The corrupt file is left in place so the user can
        inspect it.
        """
        key = _cache_key(
            model=model,
            prompt=prompt,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_key_parts=extra_key_parts,
        )
        path = self._path_for(key)
        if not path.is_file():
            self._misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = RawResponse(**payload)
        except (OSError, json.JSONDecodeError, ValidationError, TypeError):
            self._misses += 1
            return None
        self._hits += 1
        return response

    def put(
        self,
        response: RawResponse,
        *,
        prompt: str,
        seed: int,
        temperature: float,
        max_tokens: int,
        extra_key_parts: tuple[str, ...] = (),
    ) -> None:
        """Write ``response`` to disk under the canonical key.

        The cache key is derived from ``response.model`` + the supplied
        ``prompt`` / ``seed`` / ``temperature`` / ``max_tokens``. The
        prompt cannot be reconstructed from the :class:`RawResponse`
        alone, so callers must pass it explicitly.

        The write is atomic AND safe against concurrent writers. Payload
        is staged via :func:`tempfile.mkstemp` (which produces a unique
        filename inside the shard directory), ``os.fsync``-ed, then
        ``os.replace``-d into ``<key>.json``. Two concurrent writers
        targeting the same key will both succeed — the last
        ``os.replace`` wins, and the loser leaves no debris (its temp
        file was already consumed by ``os.replace``). After the rename
        the parent directory is best-effort fsynced so the entry is
        visible across a power-loss boundary on filesystems that
        support it. ``OSError`` from the parent-dir fsync is logged at
        debug level and swallowed because some filesystems (NFS,
        Windows) do not support that call.
        """
        key = _cache_key(
            model=response.model,
            prompt=prompt,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_key_parts=extra_key_parts,
        )
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            response.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        # ``tempfile.mkstemp`` returns a unique name within the shard
        # directory, so two concurrent writers do not clobber each
        # other's staging file. Both succeed; the last ``os.replace``
        # wins. ``prefix=key + "."`` keeps debris visually grouped if a
        # process dies between mkstemp and replace.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f"{key}.",
            suffix=".tmp.json",
        )
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_name, path)
        # fsync the parent directory so the rename survives a crash on
        # filesystems that journal data separately from metadata. This
        # call is unsupported on NFS / Windows — swallow ``OSError`` so
        # the cache works there too.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError as exc:
            logger.debug("parent dir open for fsync failed: %s", exc)
        else:
            try:
                os.fsync(dir_fd)
            except OSError as exc:
                logger.debug("parent dir fsync failed: %s", exc)
            finally:
                os.close(dir_fd)
        self._writes += 1

    def stats(self) -> dict[str, int]:
        """Return cache statistics for the lifetime of this instance."""
        return {"hits": self._hits, "misses": self._misses, "writes": self._writes}
