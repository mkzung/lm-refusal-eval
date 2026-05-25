"""Tests for the :class:`~lre.judge.Judge` Protocol.

F-R3-P2-10: third-party custom judges must be able to satisfy the
protocol structurally — no inheritance, no decorator, no registration.
The runner's ``ajudge_responses`` accepts any object exposing the
``label`` / ``alabel`` / ``kind`` triple.

F-R3-P2-17: ``Judge.kind`` is a runtime-enforced literal. The static
type-checker can reject ``Literal['something']`` mismatches in code we
ship, but third-party judges written without mypy still need a clear
runtime error.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lre.runner import _resolve_judge, ajudge_responses
from lre.state import RawResponse, RefusalLabel


def _make_responses(n: int = 3) -> list[RawResponse]:
    return [
        RawResponse(
            prompt_id=f"p{i}",
            model="m",
            output="content",
            generation_seconds=0.01,
            timestamp=0,
            seed=0,
        )
        for i in range(n)
    ]


def test_third_party_judge_via_ajudge_responses(
    fake_judge_factory: Any,
) -> None:
    """F-R3-P2-10: a custom judge satisfies the Protocol structurally."""
    judge = fake_judge_factory(refused=True, confidence=0.42)
    assert judge.kind == "custom"
    responses = _make_responses(3)
    labels = asyncio.run(ajudge_responses(responses, judge=judge))
    assert len(labels) == 3
    assert all(isinstance(lbl, RefusalLabel) for lbl in labels)
    assert all(lbl.refused for lbl in labels)
    assert all(lbl.heuristic_score == 0.42 for lbl in labels)


def test_third_party_judge_can_return_complied(
    fake_judge_factory: Any,
) -> None:
    """Custom judge that always returns ``refused=False`` — still valid."""
    judge = fake_judge_factory(refused=False)
    responses = _make_responses(2)
    labels = asyncio.run(ajudge_responses(responses, judge=judge))
    assert all(not lbl.refused for lbl in labels)


def test_resolve_judge_runtime_rejects_unknown_kind() -> None:
    """F-R3-P2-17: a judge whose ``kind`` is not one of the documented
    literals must be rejected with a clear ValueError, not silently
    accepted.
    """

    class BadJudge:
        kind = "wat"

        def label(
            self, prompt_id: str, model: str, response_text: str
        ) -> RefusalLabel:  # pragma: no cover — never reached
            raise NotImplementedError

        async def alabel(
            self, prompt_id: str, model: str, response_text: str
        ) -> RefusalLabel:  # pragma: no cover — never reached
            raise NotImplementedError

    with pytest.raises(ValueError, match=r"Judge\.kind"):
        _resolve_judge(judge=BadJudge(), kind="rule", llm_judge=None)


def test_resolve_judge_accepts_documented_kinds(
    fake_judge_factory: Any,
) -> None:
    """All documented ``kind`` values are accepted (the current implementation: includes 'manual')."""
    for kind in ("rule", "llm", "manual", "custom"):

        class _J:
            pass

        j = _J()
        j.kind = kind  # type: ignore[attr-defined]

        def label(
            self: Any, prompt_id: str, model: str, response_text: str
        ) -> RefusalLabel:  # pragma: no cover
            return RefusalLabel(
                prompt_id=prompt_id,
                model=model,
                refused=False,
                heuristic_score=0.0,
                judge="manual",
            )

        async def alabel(
            self: Any, prompt_id: str, model: str, response_text: str
        ) -> RefusalLabel:  # pragma: no cover
            return label(self, prompt_id, model, response_text)

        _J.label = label  # type: ignore[attr-defined]
        _J.alabel = alabel  # type: ignore[attr-defined]
        # Should not raise.
        resolved = _resolve_judge(judge=j, kind="rule", llm_judge=None)
        assert resolved is j
