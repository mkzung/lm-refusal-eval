"""Tests for :class:`lre.models.openai_api.OpenAIClient`.

We use :class:`httpx.MockTransport` to stub out the network — no real
calls are made — and patch :class:`httpx.AsyncClient` so the adapter
picks up the mock transport. The patching is contained in a context
manager so test isolation is preserved even on failures.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lre.models.openai_api import OpenAIClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _patch_async_client(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[], None]:
    """Replace ``httpx.AsyncClient`` with one that uses our mock transport.

    Returns a callable that restores the original class.
    """
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]

    def restore() -> None:
        httpx.AsyncClient = real  # type: ignore[misc]

    return restore


def test_openai_client_parses_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(
            model="gpt-test", api_key="sk-fake", base_url="https://example.test/v1"
        )
        result = _run(client.generate("prompt", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()
    assert result.output == "Hello!"
    assert result.model == "gpt-test"
    assert result.generation_seconds >= 0.0


def test_openai_client_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")
        # Patch asyncio.sleep so the retry backoff doesn't slow tests.
        import lre.models.openai_api as mod

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


def test_openai_client_fails_after_repeated_500s() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1", max_retries=3)
        import lre.models.openai_api as mod

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


def test_openai_client_raises_on_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Server returns a top-level JSON array, which the adapter rejects.
        return httpx.Response(200, json=[1, 2, 3])

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")
        with pytest.raises(TypeError):
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()


def test_openai_client_empty_choices_yields_empty_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")
        result = _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()
    assert result.output == ""


def test_openai_client_propagates_4xx_other_than_429() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # 400 is a generic client error; raises HTTPStatusError (not auth).
        return httpx.Response(400, json={"error": "bad request"})

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")
        with pytest.raises(httpx.HTTPStatusError):
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()


def test_openai_client_raises_authentication_error_on_401() -> None:
    """F-R3-P1-5: 401 must surface as a non-retryable AuthenticationError,
    not a raw httpx.HTTPStatusError with stack trace.
    """
    from lre.models.openai_api import AuthenticationError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")
        with pytest.raises(AuthenticationError) as exc_info:
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
        assert "401" in str(exc_info.value)
    finally:
        restore()


def test_openai_client_raises_authentication_error_on_403() -> None:
    """F-R3-P1-5: 403 (forbidden) is also non-retryable auth failure."""
    from lre.models.openai_api import AuthenticationError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    restore = _patch_async_client(handler)
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")
        with pytest.raises(AuthenticationError):
            _run(client.generate("p", temperature=0.0, max_tokens=64, seed=0))
    finally:
        restore()


def test_openai_client_reuses_async_client_across_calls() -> None:
    """F-R2-P2-10: a single OpenAIClient must reuse one httpx.AsyncClient
    across multiple ``generate`` calls.
    """
    construct_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        construct_calls["n"] += 1
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]
    try:
        client = OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1")

        async def go() -> None:
            await client.generate("a", temperature=0.0, max_tokens=8, seed=0)
            await client.generate("b", temperature=0.0, max_tokens=8, seed=0)
            await client.generate("c", temperature=0.0, max_tokens=8, seed=0)
            await client.aclose()

        _run(go())
    finally:
        httpx.AsyncClient = real  # type: ignore[misc]
    assert construct_calls["n"] == 1


def test_openai_client_works_as_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    restore = _patch_async_client(handler)
    try:

        async def go() -> None:
            async with OpenAIClient(model="m", api_key="k", base_url="https://x.test/v1") as c:
                await c.generate("p", temperature=0.0, max_tokens=8, seed=0)
                assert c._http is not None
            assert c._http is None

        _run(go())
    finally:
        restore()


# ---------------------------------------------------------------------------
# v0.8 (P1-6, P1-20): jitter is seeded; httpx client respects configured
# Limits + Timeout
# ---------------------------------------------------------------------------


def test_openai_jitter_is_seeded_for_byte_identical_retries() -> None:
    """Two clients with the same ``jitter_seed`` produce the same backoff sequence."""
    a = OpenAIClient(model="m", api_key="k", jitter_seed=99)
    b = OpenAIClient(model="m", api_key="k", jitter_seed=99)
    seq_a = [a._jitter_rng.uniform(0.0, 1.0) for _ in range(5)]
    seq_b = [b._jitter_rng.uniform(0.0, 1.0) for _ in range(5)]
    assert seq_a == seq_b


def test_openai_httpx_client_uses_configured_limits_and_timeout() -> None:
    """v0.8 (P1-20): the constructed ``httpx.AsyncClient`` is built with
    the configured connection-pool limits and granular timeouts.

    Captures the kwargs the adapter passes into ``httpx.AsyncClient(...)``
    via a factory hook.
    """
    captured: dict[str, Any] = {}
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        captured.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )
        )
        return real(*args, **kwargs)

    httpx.AsyncClient = factory  # type: ignore[misc, assignment]
    try:

        async def go() -> None:
            client = OpenAIClient(
                model="m",
                api_key="k",
                base_url="https://x.test/v1",
                max_connections=72,
                max_keepalive_connections=18,
                connect_timeout=4.0,
                read_timeout=33.0,
                write_timeout=6.0,
            )
            await client.generate("p", temperature=0.0, max_tokens=8, seed=0)
            await client.aclose()

        _run(go())
    finally:
        httpx.AsyncClient = real  # type: ignore[misc]

    limits = captured.get("limits")
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 72
    assert limits.max_keepalive_connections == 18
    timeout = captured.get("timeout")
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 4.0
    assert timeout.read == 33.0
    assert timeout.write == 6.0
