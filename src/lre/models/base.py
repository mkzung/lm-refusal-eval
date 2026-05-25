"""Abstract :class:`ModelClient` protocol that every adapter must satisfy.

We deliberately use a :class:`~typing.Protocol` rather than an ABC so that
third-party clients (mocks, stubs, lightweight wrappers) can satisfy the
interface structurally — no inheritance required. This keeps the harness
adapter-agnostic, which matters for downstream scaling-laws studies that
need to swap in dozens of model families with minimal ceremony.

Note: this protocol is **not** decorated with ``@runtime_checkable``.
``isinstance(x, ModelClient)`` for a Protocol with a non-method
``name: str`` attribute is unreliable across Python 3.10 / 3.11 / 3.12 —
the runtime check only inspects method presence, not attribute presence,
which leads to false positives. The harness validates clients statically
(``mypy --strict``) and via the dependency-injection sites in
``lre.runner``; if a runtime check is needed, prefer ``hasattr(x, "name")
and hasattr(x, "generate")``.
"""

from __future__ import annotations

from typing import Protocol

from lre.state import RawResponse


class ModelClient(Protocol):
    """Minimal interface every model adapter must expose.

    Implementations must be safe to call concurrently from an ``asyncio``
    event loop and should perform their own thread-pool offloading for any
    blocking I/O (e.g. ``transformers`` inference).
    """

    name: str

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> RawResponse:
        """Generate one completion and return it as a :class:`RawResponse`.

        Implementations should populate ``prompt_id`` with an opaque value
        when the caller does not supply one — the runner overrides it.
        """
        ...
