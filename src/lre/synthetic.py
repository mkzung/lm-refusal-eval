"""Deterministic synthetic ``ModelClient`` for tests and the demo.

Real model adapters require either GPU weights (``HFLocalClient``) or paid
API keys (``OpenAIClient`` / ``AnthropicClient``). Neither is appropriate
for unit tests or the ``lre demo`` command. This module fills the gap with
a client that returns canned refusal/compliance text based on a hash of
``(prompt_id, seed)``, allowing the test suite — and any researcher
exploring the harness — to exercise the full pipeline offline.

The hash is plain Python (``hashlib.sha256``) so output is identical
across platforms and Python versions; ``hash()`` would not be.
"""

from __future__ import annotations

import hashlib
import time

from lre.state import RawResponse

# Canonical templates. The refusal template is intentionally rich in
# trigger phrases so :class:`lre.judge.RuleBasedJudge` reliably flags it;
# the compliance template avoids every documented trigger.
_REFUSAL_TEMPLATE = (
    "I'm sorry, but I can't help with that request. It goes against my "
    "guidelines, and I'm not able to provide instructions of this kind. "
    "If you have a related but lawful question, I'd be happy to help."
)

_COMPLIANCE_TEMPLATE = (
    "Sure — here's a concise answer. The topic involves several moving "
    "parts; the most relevant ones for a working understanding are A, B, "
    "and C. If you'd like a deeper dive into any of these, let me know."
)


def _refusal_score(text: str, seed: int) -> float:
    """Return a deterministic float in ``[0, 1)`` derived from ``text`` and ``seed``.

    Both arguments are mixed into a SHA-256 digest; the first eight bytes
    are interpreted as an unsigned 64-bit integer and normalized to
    ``[0, 1)``. ``text`` is the *prompt body* — :meth:`FakeModelClient.generate`
    passes the raw prompt string, not the prompt id, so two prompts with
    the same id but different text still receive different scores.
    """
    digest = hashlib.sha256(f"{text}|{seed}".encode()).digest()
    # Take the first 8 bytes as an unsigned 64-bit int, then normalize.
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


class FakeModelClient:
    """Synthetic deterministic adapter.

    Parameters
    ----------
    name:
        Identifier reported in :class:`RawResponse.model` (and used by the
        scaling-table pivot).
    refusal_rate:
        Probability — in expectation — that any single prompt is refused.
        Must lie in ``[0.0, 1.0]``.
    seed:
        Base seed mixed into the per-prompt hash, so the same client at
        the same ``seed`` always yields the same outputs.
    latency_seconds:
        Synthetic generation-time stamped onto each :class:`RawResponse`.
        Real models would compute this from the wall clock; the fake
        client uses a fixed value so report tests are byte-stable.
    """

    def __init__(
        self,
        name: str = "fake-1b",
        *,
        refusal_rate: float = 0.5,
        seed: int = 42,
        latency_seconds: float = 0.012,
    ) -> None:
        if not 0.0 <= refusal_rate <= 1.0:
            msg = f"refusal_rate must be in [0, 1]; got {refusal_rate!r}"
            raise ValueError(msg)
        self.name = name
        self._refusal_rate = refusal_rate
        self._base_seed = seed
        self._latency = latency_seconds

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> RawResponse:
        # Mix the caller-supplied seed with the per-client seed so callers
        # can rerun the same client deterministically by varying ``seed``.
        effective_seed = (seed * 2654435761 + self._base_seed) & 0xFFFFFFFF
        score = _refusal_score(prompt, effective_seed)
        text = _REFUSAL_TEMPLATE if score < self._refusal_rate else _COMPLIANCE_TEMPLATE
        return RawResponse(
            prompt_id="",
            model=self.name,
            output=text,
            generation_seconds=self._latency,
            timestamp=int(time.time()),
            seed=seed,
        )
