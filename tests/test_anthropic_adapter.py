"""Tests for :class:`lre.models.anthropic_api.AnthropicClient`.

We stub the network via :class:`httpx.MockTransport`, patching
``httpx.AsyncClient`` inside a context manager. No real API calls are
made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lre.models.anthropic_api import AnthropicClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _patch_async_client(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[], None]:
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]

    def restore() -> None:
        httpx.AsyncClient = real  # type: ignore[misc]

    return restore


def test_anthropic_client_parses_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Verify the Anthropic-specific headers are present.
        assert request.headers["x-api-key"] == "sk-test"
        assert request.headers["anthropic-version"] == AnthropicClient.DEFAULT_API_VERSION
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
                "model": "claude-test",
                "stop_reason": "end_turn",
            },
        )

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(
            model="claude-test", api_key="sk-test", base_url="https://example.test"
        )
        result = _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()
    assert result.output == "Hi there!"
    assert result.model == "claude-test"


def test_anthropic_client_joins_multiple_text_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Hello, "},
                    {"type": "tool_use", "id": "tu", "input": {}},  # ignored
                    {"type": "text", "text": "world."},
                ]
            },
        )

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        result = _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()
    assert result.output == "Hello, world."


def test_anthropic_client_retries_on_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "limit"})
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "ok"}]},
        )

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        import lre.models.anthropic_api as mod

        real_sleep = mod.asyncio.sleep

        async def fast_sleep(_d: float) -> None:
            return None

        mod.asyncio.sleep = fast_sleep  # type: ignore[assignment]
        try:
            result = _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
        finally:
            mod.asyncio.sleep = real_sleep  # type: ignore[assignment]
    finally:
        restore()
    assert result.output == "ok"
    assert calls["n"] == 2


def test_anthropic_client_fails_after_repeated_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test", max_retries=3)
        import lre.models.anthropic_api as mod

        real_sleep = mod.asyncio.sleep

        async def fast_sleep(_d: float) -> None:
            return None

        mod.asyncio.sleep = fast_sleep  # type: ignore[assignment]
        try:
            with pytest.raises(httpx.HTTPStatusError):
                _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
        finally:
            mod.asyncio.sleep = real_sleep  # type: ignore[assignment]
    finally:
        restore()


def test_anthropic_client_rejects_non_dict_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        with pytest.raises(TypeError):
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()


def test_anthropic_client_propagates_other_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # 400 (generic client error) — not auth, so propagates as HTTPStatusError.
        return httpx.Response(400, json={"error": "bad request"})

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        with pytest.raises(httpx.HTTPStatusError):
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()


def test_anthropic_client_raises_authentication_error_on_401() -> None:
    """F-R3-P1-5: 401 must surface as a non-retryable AuthenticationError,
    not a raw httpx.HTTPStatusError with stack trace.
    """
    from lre.models.anthropic_api import AuthenticationError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        with pytest.raises(AuthenticationError) as exc_info:
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
        assert "401" in str(exc_info.value)
    finally:
        restore()


def test_anthropic_client_raises_authentication_error_on_403() -> None:
    """F-R3-P1-5: 403 (forbidden) is also non-retryable auth failure."""
    from lre.models.anthropic_api import AuthenticationError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        with pytest.raises(AuthenticationError):
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()


def test_anthropic_client_empty_content_yields_empty_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": []})

    restore = _patch_async_client(handler)
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")
        result = _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()
    assert result.output == ""


def test_anthropic_client_reuses_async_client_across_calls() -> None:
    """F-R2-P2-10: a single AnthropicClient must reuse one
    httpx.AsyncClient across multiple ``generate`` calls. We mock the
    httpx.AsyncClient constructor and assert it is invoked exactly once
    even after several requests.
    """
    construct_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        construct_calls["n"] += 1
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]
    try:
        client = AnthropicClient(model="m", api_key="k", base_url="https://x.test")

        async def go() -> None:
            await client.generate("a", temperature=0.0, max_tokens=8, seed=0)
            await client.generate("b", temperature=0.0, max_tokens=8, seed=0)
            await client.generate("c", temperature=0.0, max_tokens=8, seed=0)
            await client.aclose()

        _run(go())
    finally:
        httpx.AsyncClient = real  # type: ignore[misc]
    assert construct_calls["n"] == 1, (
        f"expected exactly one AsyncClient construction, got {construct_calls['n']}"
    )


def test_anthropic_client_works_as_context_manager() -> None:
    """``async with AnthropicClient(...) as c:`` closes the underlying
    httpx client on exit.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    restore = _patch_async_client(handler)
    try:

        async def go() -> None:
            async with AnthropicClient(model="m", api_key="k", base_url="https://x.test") as c:
                await c.generate("p", temperature=0.0, max_tokens=8, seed=0)
                assert c._http is not None
            assert c._http is None

        _run(go())
    finally:
        restore()


# ---------------------------------------------------------------------------
# the current implementation (P1-6, P1-20): jitter is seeded; httpx client respects configured
# Limits + Timeout
# ---------------------------------------------------------------------------


def test_anthropic_jitter_is_seeded_for_byte_identical_retries() -> None:
    """Two clients with the same ``jitter_seed`` produce the same
    decorrelated-backoff sequence.

    An earlier iteration ``random.uniform`` used the module-level RNG, so retries
    were non-reproducible — silently breaking the byte-identity claim
    for any run that ever hit a 429.
    """
    a = AnthropicClient(model="m", api_key="k", jitter_seed=42)
    b = AnthropicClient(model="m", api_key="k", jitter_seed=42)
    seq_a = [a._jitter_rng.uniform(0.0, 1.0) for _ in range(5)]
    seq_b = [b._jitter_rng.uniform(0.0, 1.0) for _ in range(5)]
    assert seq_a == seq_b
    # Different seed → different sequence.
    c = AnthropicClient(model="m", api_key="k", jitter_seed=1)
    seq_c = [c._jitter_rng.uniform(0.0, 1.0) for _ in range(5)]
    assert seq_c != seq_a


def test_anthropic_httpx_client_uses_configured_limits_and_timeout() -> None:
    """the constructed ``httpx.AsyncClient`` is built with
    the configured connection-pool limits and granular timeouts.

    Captures the kwargs the adapter passes into ``httpx.AsyncClient(...)``
    via a factory hook, so the assertion does not depend on the
    underlying httpx version's internal attribute layout.
    """
    captured: dict[str, Any] = {}
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        captured.update(kwargs)
        # Inject a MockTransport so no real network is touched.
        kwargs["transport"] = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})
        )
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]
    try:

        async def go() -> None:
            client = AnthropicClient(
                model="m",
                api_key="k",
                base_url="https://x.test",
                max_connections=64,
                max_keepalive_connections=16,
                connect_timeout=5.0,
                read_timeout=30.0,
                write_timeout=7.0,
            )
            await client.generate("p", temperature=0.0, max_tokens=8, seed=0)
            await client.aclose()

        _run(go())
    finally:
        httpx.AsyncClient = real  # type: ignore[misc]

    limits = captured.get("limits")
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 64
    assert limits.max_keepalive_connections == 16
    timeout = captured.get("timeout")
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read == 30.0
    assert timeout.write == 7.0
