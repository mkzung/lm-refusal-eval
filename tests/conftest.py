"""Shared pytest fixtures (F-R3-P2-11 / NEW-R3-4).

The httpx-mocking helpers in the adapter test modules previously rolled
their own ``try/finally`` block to restore ``httpx.AsyncClient`` after a
test. That works for the happy path but leaks the monkeypatch into
neighbouring tests if a Ctrl-C arrives between ``factory =`` and
``finally:``. Centralising the pattern behind a pytest fixture lets
pytest's normal teardown machinery clean up even on cancellation.

Three fixtures are exposed:

* :func:`mock_httpx_async_client` — yields a setup helper that installs
  an :class:`httpx.MockTransport`-backed factory in place of
  :class:`httpx.AsyncClient`. The original class is restored on
  teardown via ``monkeypatch.undo``.
* :func:`tmp_suite_jsonl` — yields a builder that writes a temp JSONL
  suite from a list of ``Prompt`` payload dicts and returns the path.
* :func:`fake_judge_factory` — yields a builder that creates a minimal
  custom-``kind`` :class:`Judge` returning a fixed verdict.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from lre.state import RefusalLabel


@pytest.fixture
def mock_httpx_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[Callable[[httpx.Request], httpx.Response]], None]]:
    """Patch ``httpx.AsyncClient`` to route every request through a mock.

    Usage::

        def test_x(mock_httpx_async_client):
            def handler(req): return httpx.Response(200, json={...})
            mock_httpx_async_client(handler)
            ...

    The patch is registered via ``monkeypatch``, so pytest guarantees it
    is undone after the test — even on failures or interrupts.
    """
    real = httpx.AsyncClient
    installed = {"value": False}

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        transport = httpx.MockTransport(handler)

        def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        installed["value"] = True

    yield install
    # No explicit teardown — monkeypatch.undo handles restoration even
    # if the install() helper was never called.


@pytest.fixture
def tmp_suite_jsonl(tmp_path: Path) -> Callable[[list[dict[str, Any]], str], Path]:
    """Write a JSONL suite from a list of prompt-payload dicts.

    Returns a builder so individual tests can populate the file with the
    shape they need. The file is created under ``tmp_path`` and cleaned
    up by pytest's normal tmp-dir machinery.
    """

    def build(rows: list[dict[str, Any]], filename: str = "suite.jsonl") -> Path:
        path = tmp_path / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    return build


@pytest.fixture
def fake_judge_factory() -> Callable[..., Any]:
    """Build a minimal third-party :class:`~lre.judge.Judge`.

    The returned class declares ``kind='custom'`` and emits a
    :class:`RefusalLabel` with the requested ``refused`` verdict. Used
    by F-R3-P2-10 to confirm ``ajudge_responses`` accepts arbitrary
    custom judges via the structural protocol.
    """

    def make(refused: bool = True, confidence: float = 0.5) -> Any:
        class _FakeJudge:
            kind = "custom"

            def label(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
                return RefusalLabel(
                    prompt_id=prompt_id,
                    model=model,
                    refused=refused,
                    heuristic_score=confidence,
                    judge="manual",
                )

            async def alabel(self, prompt_id: str, model: str, response_text: str) -> RefusalLabel:
                return self.label(prompt_id, model, response_text)

        return _FakeJudge()

    return make
