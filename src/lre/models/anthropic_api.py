"""Anthropic Messages API adapter via ``httpx``.

We keep this as a thin wrapper over the documented public endpoint instead
of pulling in the official SDK — fewer transitive deps, and the request
surface for our needs is small. Refer to the official SDK for production
use cases that need streaming, tool use, vision, etc.

Determinism caveat: Anthropic does not currently expose a ``seed``
parameter, so we record the requested seed in :class:`RawResponse.seed`
purely as provenance metadata. Treat API runs as non-deterministic.

Connection reuse
----------------
A single long-lived :class:`httpx.AsyncClient` is lazily constructed on
the first :meth:`generate` call and reused for every subsequent request.
This avoids the per-request TLS handshake cost that the previous
short-lived ``async with httpx.AsyncClient(...)`` pattern imposed. Call
:meth:`aclose` (or use the client as an ``async with`` context manager)
to release the underlying TCP/TLS resources when done.
"""

from __future__ import annotations

import asyncio
import random
import time
from types import TracebackType
from typing import Any

import click
import httpx

from lre.state import RawResponse


class AuthenticationError(click.UsageError):
    """Raised when the API returned 401/403 — credentials are missing or invalid.

    Subclasses :class:`click.UsageError` so the CLI converts it to a clean
    exit (no traceback). The runner explicitly does not catch this class;
    auth failures are non-retryable and must propagate.
    """


class AnthropicClient:
    """Thin async client over the Anthropic ``/v1/messages`` endpoint."""

    DEFAULT_API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        api_version: str = DEFAULT_API_VERSION,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_connections: int = 128,
        max_keepalive_connections: int = 32,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        write_timeout: float = 10.0,
        retry_base_delay: float = 1.0,
        jitter_seed: int = 0,
    ) -> None:
        self.name = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._retry_base_delay = retry_base_delay
        self._http: httpx.AsyncClient | None = None
        self._http_lock = asyncio.Lock()
        # v0.8 (P1-6): decorrelated-exponential-backoff jitter must be
        # seeded so two ``generate()`` calls with the same seed produce
        # the same jitter sequence. Pre-v0.8 the calls used
        # ``random.uniform`` against the module-level shared RNG, which
        # silently broke the byte-identity claim for runs that ever hit
        # the retry path.
        self._jitter_rng = random.Random(jitter_seed)

    async def _get_http(self) -> httpx.AsyncClient:
        """Lazily construct the shared :class:`httpx.AsyncClient`.

        Guarded by a lock so the very first concurrent fan-out does not
        race and construct two clients. After the first request, the
        common path is the unsynchronized fast-return.
        """
        if self._http is not None:
            return self._http
        async with self._http_lock:
            if self._http is None:
                # Explicit connection-pool limits prevent unbounded
                # socket creation under high fan-out (a 100-prompt
                # ``lre run --max-concurrent 64`` previously opened up
                # to 64 short-lived sockets — the limits below cap and
                # keep-alive them).
                limits = httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive_connections,
                )
                # Granular timeout splits the single ``timeout`` value
                # into connect / read / write so a slow-reading endpoint
                # cannot starve the pool of new connections.
                timeout = httpx.Timeout(
                    connect=self._connect_timeout,
                    read=self._read_timeout,
                    write=self._write_timeout,
                    pool=None,
                )
                self._http = httpx.AsyncClient(timeout=timeout, limits=limits)
            return self._http

    async def aclose(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient`.

        Idempotent. After :meth:`aclose`, the next :meth:`generate` call
        constructs a fresh client.
        """
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> AnthropicClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> RawResponse:
        url = f"{self._base_url}/v1/messages"
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        body = await self._post_with_retries(url, payload, headers)
        elapsed = time.perf_counter() - start
        text = self._extract_text(body)
        return RawResponse(
            prompt_id="",
            model=self.name,
            output=text,
            generation_seconds=elapsed,
            timestamp=int(time.time()),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _post_with_retries(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        base = self._retry_base_delay
        last_exc: Exception | None = None
        client = await self._get_http()
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_exc = exc
            else:
                if response.status_code < 400:
                    result = response.json()
                    if not isinstance(result, dict):
                        msg = "Anthropic response is not a JSON object"
                        raise TypeError(msg)
                    return result
                # Auth failures (401/403) are non-retryable: the credential
                # is missing or rejected, and no amount of backoff will
                # change that. Raise a clean click.UsageError-derived
                # exception so the CLI surfaces a one-line error.
                if response.status_code in (401, 403):
                    msg = (
                        f"Anthropic authentication failed (HTTP {response.status_code}). "
                        "Check ANTHROPIC_API_KEY (or --api-key-env)."
                    )
                    raise AuthenticationError(msg)
                if response.status_code not in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if attempt < self._max_retries:
                # Decorrelated exponential backoff with jitter: prevents
                # the thundering-herd retry storm a synchronised fleet
                # of workers triggers under a transient 429. Jitter
                # comes from the per-instance seeded RNG so a run with
                # a fixed seed reproduces the same backoff sequence.
                delay = base * (2 ** (attempt - 1)) + self._jitter_rng.uniform(0.0, base)
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        parts = body.get("content") or []
        if not parts:
            return ""
        chunks: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
