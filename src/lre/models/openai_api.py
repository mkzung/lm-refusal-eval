"""OpenAI-compatible chat-completions adapter via ``httpx``.

The adapter is endpoint-agnostic: any provider that exposes an
OpenAI-compatible ``/v1/chat/completions`` route (vLLM, llama.cpp server,
TogetherAI, Groq, etc.) works by setting ``base_url`` accordingly.

Determinism caveat: the OpenAI API ``seed`` parameter is best-effort, not
contractual. We pass it through, but downstream researchers should treat
API runs as not byte-identically reproducible. Local backends like vLLM
respect the seed exactly when ``temperature=0``.

Connection reuse
----------------
A single long-lived :class:`httpx.AsyncClient` is lazily constructed on
the first :meth:`generate` call and reused thereafter. Use as an
``async with`` context manager — or call :meth:`aclose` — to release
the underlying TCP/TLS pool when done.
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
    exit with code 2 (no traceback). The runner explicitly does not catch
    this class; auth failures are non-retryable and must propagate.
    """


class OpenAIClient:
    """Thin async client over the OpenAI chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
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
        # seeded jitter RNG — see AnthropicClient for the
        # full rationale on byte-identical retries.
        self._jitter_rng = random.Random(jitter_seed)

    async def _get_http(self) -> httpx.AsyncClient:
        """Lazily construct the shared :class:`httpx.AsyncClient`."""
        if self._http is not None:
            return self._http
        async with self._http_lock:
            if self._http is None:
                # See AnthropicClient._get_http for the rationale on
                # explicit connection-pool limits + granular timeouts.
                limits = httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive_connections,
                )
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

    async def __aenter__(self) -> OpenAIClient:
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
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
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
                        msg = "OpenAI response is not a JSON object"
                        raise TypeError(msg)
                    return result
                # Auth failures (401/403) are non-retryable: the credential
                # is missing or rejected, and no amount of backoff will
                # change that. Raise a clean click.UsageError-derived
                # exception so the CLI surfaces a one-line error.
                if response.status_code in (401, 403):
                    msg = (
                        f"OpenAI authentication failed (HTTP {response.status_code}). "
                        "Check OPENAI_API_KEY (or --api-key-env)."
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
                # Exponential backoff with seeded jitter — same seam as
                # AnthropicClient so a fixed-seed run reproduces the
                # same retry timing.
                delay = base * (2 ** (attempt - 1)) + self._jitter_rng.uniform(0.0, base)
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return str(content) if content is not None else ""
