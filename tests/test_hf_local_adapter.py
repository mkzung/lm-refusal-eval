"""Tests for :class:`lre.models.hf_local.HFLocalClient`.

We never download model weights or import torch in these tests. The
adapter is constructed and inspected purely at the Python level. The
``generate`` path is exercised by monkeypatching ``_load`` and
``_generate_sync`` so we never touch the GPU or the network.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from lre.models.hf_local import HFLocalClient


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_hf_client_init_does_not_import_torch() -> None:
    """Constructing the client must not trigger any heavyweight import.
    The whole point of the lazy-load strategy is that researchers who
    only use the API adapters don't pay the torch import cost.
    """
    # Wipe any cached imports so the assertion is meaningful.
    for name in list(sys.modules):
        if name.startswith(("torch", "transformers")):
            del sys.modules[name]
    client = HFLocalClient(model_id="fake/model", name="local-test")
    assert client.name == "local-test"
    assert client.model_id == "fake/model"
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_hf_client_generate_raises_helpful_import_error_when_deps_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When transformers / torch are not installed, calling ``generate``
    must raise an ``ImportError`` that names the optional extra to install.
    """
    # Make import torch / from transformers explode regardless of whether
    # transformers happens to be installed in the test env.
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )  # type: ignore[union-attr]

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("torch",) or name.startswith("transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    client = HFLocalClient(model_id="fake/model")
    with pytest.raises(ImportError, match="hf"):
        _run(client.generate("hello", temperature=0.0, max_tokens=8, seed=0))


def test_hf_client_generate_calls_lazy_load_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock ``_generate_sync`` so we can verify the async wrapper without
    needing the actual model weights.
    """
    client = HFLocalClient(model_id="fake/model", name="local-mock")

    def fake_generate_sync(
        self: HFLocalClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        return f"echo:{prompt}|t={temperature}|m={max_tokens}|s={seed}"

    monkeypatch.setattr(HFLocalClient, "_generate_sync", fake_generate_sync)
    result = _run(client.generate("hi", temperature=0.1, max_tokens=4, seed=7))
    assert result.output == "echo:hi|t=0.1|m=4|s=7"
    assert result.model == "local-mock"
    assert result.generation_seconds >= 0.0


def test_hf_client_defaults_name_to_model_id() -> None:
    """Base-model (``use_chat_template=False``) default name is plain ``model_id``.

    The ``@chat`` suffix only kicks in when the default chat template is
    on, so a researcher who explicitly opts out of chat-templating keeps
    the bare model id in reports and caches.
    """
    client = HFLocalClient(model_id="org/some-repo", use_chat_template=False)
    assert client.name == "org/some-repo"


def test_hf_client_defaults_name_appends_chat_suffix_when_template_on() -> None:
    """Default chat-template path bakes ``@chat`` into the effective name.

    F-R4-P1-2: without this, two clients loading the same weights with
    different ``use_chat_template`` flags would collide on the cache
    key. The effective name participates in ``model`` so the existing
    SHA-256 key derivation distinguishes them automatically.
    """
    client = HFLocalClient(model_id="org/some-repo")
    assert client.use_chat_template is True
    assert client.name == "org/some-repo@chat"


def test_hf_client_explicit_name_overrides_chat_suffix() -> None:
    """User-supplied ``name=`` always wins over the auto-suffix default."""
    client = HFLocalClient(model_id="org/some-repo", name="manual-name")
    assert client.name == "manual-name"


# ---------------------------------------------------------------------------
# v0.5: chat-template support
# ---------------------------------------------------------------------------


class _FakeTokenizerWithChatTemplate:
    """Test double for a tokenizer that implements ``apply_chat_template``."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.applied_messages: list[list[dict[str, str]]] = []
        self.tokenized_prompts: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # Record what the adapter sent us so the test can introspect.
        self.applied_messages.append(messages)
        # Mimic the canonical instruct-template envelope.
        return f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>"

    def __call__(self, prompt: str, *, return_tensors: str = "pt") -> Any:
        self.tokenized_prompts.append(prompt)

        class _Inputs(dict[str, Any]):
            def to(self_inner, device: str) -> _Inputs:
                return self_inner

        # Minimal shape: a 1xN "input_ids" tensor stub.
        return _Inputs(input_ids=_FakeTensor([list(range(len(prompt)))]))

    def decode(self, ids: Any, *, skip_special_tokens: bool = True) -> str:
        return "decoded"


class _FakeTokenizerNoChatTemplate:
    """Test double for a base-model tokenizer (no apply_chat_template attribute)."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.tokenized_prompts: list[str] = []

    def __call__(self, prompt: str, *, return_tensors: str = "pt") -> Any:
        self.tokenized_prompts.append(prompt)

        class _Inputs(dict[str, Any]):
            def to(self_inner, device: str) -> _Inputs:
                return self_inner

        return _Inputs(input_ids=_FakeTensor([list(range(len(prompt)))]))

    def decode(self, ids: Any, *, skip_special_tokens: bool = True) -> str:
        return "decoded-base"


class _FakeTensor:
    """Just enough of a torch tensor surface for shape + slicing inspection."""

    def __init__(self, data: list[list[int]]):
        self._data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor([[0]])


class _FakeModel:
    """Causal-LM test double."""

    def generate(self, **kwargs: Any) -> Any:
        # Return a 1x1 "tensor" — the adapter slices to the completion
        # portion and decodes via the tokenizer.
        return _FakeTensor([[0, 1, 2, 3]])

    def to(self, device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    model: Any,
) -> None:
    """Patch the imports inside :class:`HFLocalClient` to return our fakes."""
    import types

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str) -> Any:
            return tokenizer

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Any:
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    fake_transformers.AutoModelForCausalLM = _FakeAutoModelForCausalLM  # type: ignore[attr-defined]
    fake_transformers.set_seed = lambda _seed: None  # type: ignore[attr-defined]

    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch.cuda = _Cuda()  # type: ignore[attr-defined]
    fake_torch.float16 = "float16"  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_hf_local_applies_chat_template_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizerWithChatTemplate()
    model = _FakeModel()
    _install_fake_transformers(monkeypatch, tokenizer, model)
    client = HFLocalClient(
        model_id="fake/model",
        name="hf-test",
        device="cpu",
        use_chat_template=True,
    )
    _run(client.generate("hello world", temperature=0.0, max_tokens=8, seed=0))
    # The tokenizer received the chat-templated payload, not the raw prompt.
    assert tokenizer.applied_messages == [[{"role": "user", "content": "hello world"}]]
    assert tokenizer.tokenized_prompts == ["<|im_start|>user\nhello world<|im_end|>"]


def test_hf_local_skips_chat_template_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizerWithChatTemplate()
    model = _FakeModel()
    _install_fake_transformers(monkeypatch, tokenizer, model)
    client = HFLocalClient(
        model_id="fake/model",
        name="hf-test",
        device="cpu",
        use_chat_template=False,
    )
    _run(client.generate("hello world", temperature=0.0, max_tokens=8, seed=0))
    # apply_chat_template was NOT called.
    assert tokenizer.applied_messages == []
    # The raw prompt went straight to the tokenizer.
    assert tokenizer.tokenized_prompts == ["hello world"]


class _FakeTokenizerChatTemplateRaises:
    """Tokenizer that has ``apply_chat_template`` but raises on call.

    F-R4-P2-16: real-world tokenizers can expose ``apply_chat_template``
    yet have no Jinja template registered (raises ``ValueError``) or a
    malformed template (raises ``TemplateError``). The adapter must
    swallow the exception and fall back to the raw prompt, not abort an
    entire eval.
    """

    eos_token_id = 0

    def __init__(self) -> None:
        self.tokenized_prompts: list[str] = []
        self.apply_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        self.apply_calls += 1
        raise RuntimeError("no chat template registered")

    def __call__(self, prompt: str, *, return_tensors: str = "pt") -> Any:
        self.tokenized_prompts.append(prompt)

        class _Inputs(dict[str, Any]):
            def to(self_inner, device: str) -> _Inputs:
                return self_inner

        return _Inputs(input_ids=_FakeTensor([list(range(len(prompt)))]))

    def decode(self, ids: Any, *, skip_special_tokens: bool = True) -> str:
        return "decoded-fallback"


def test_hf_local_falls_back_when_apply_chat_template_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising ``apply_chat_template`` must not abort generation."""
    import logging

    tokenizer = _FakeTokenizerChatTemplateRaises()
    model = _FakeModel()
    _install_fake_transformers(monkeypatch, tokenizer, model)
    client = HFLocalClient(
        model_id="fake/raising-model",
        name="hf-test",
        device="cpu",
        use_chat_template=True,
    )
    with caplog.at_level(logging.WARNING, logger="lre.models.hf_local"):
        result = _run(client.generate("hi there", temperature=0.0, max_tokens=8, seed=0))
    assert result.output == "decoded-fallback"
    # The adapter attempted the chat template (proves the try-block ran)
    # and then forwarded the raw prompt to the tokenizer.
    assert tokenizer.apply_calls == 1
    assert tokenizer.tokenized_prompts == ["hi there"]
    # A warning was logged so the operator can spot the silent fallback.
    assert any("apply_chat_template raised" in rec.message for rec in caplog.records)


def test_hf_local_skips_chat_template_when_tokenizer_lacks_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tokenizer with no ``apply_chat_template`` falls back to raw prompts."""
    tokenizer = _FakeTokenizerNoChatTemplate()
    model = _FakeModel()
    _install_fake_transformers(monkeypatch, tokenizer, model)
    client = HFLocalClient(
        model_id="fake/model",
        name="hf-test",
        device="cpu",
        use_chat_template=True,  # default
    )
    _run(client.generate("base model prompt", temperature=0.0, max_tokens=8, seed=0))
    # No exception — the raw prompt went straight in.
    assert tokenizer.tokenized_prompts == ["base model prompt"]


def test_hf_local_default_use_chat_template_is_true() -> None:
    client = HFLocalClient(model_id="fake/model")
    assert client.use_chat_template is True


def test_hf_local_generation_lock_serialises_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.7/v0.8 (P1-20): the per-instance ``asyncio.Lock`` serialises
    concurrent ``generate()`` calls.

    Without the lock, four concurrent ``generate`` tasks would all sit
    inside ``_generate_sync`` simultaneously, interleaving
    ``transformers.set_seed`` state. With the lock, the four entries
    must produce strictly monotone ordering — each call sees the previous
    call's exit before its own entry.
    """
    import time as _real_time

    entry_times: list[float] = []
    exit_times: list[float] = []

    def fake_generate_sync(
        self: HFLocalClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        entry_times.append(_real_time.perf_counter())
        # Hold the executor briefly so overlap is observable if the lock
        # regresses.
        _real_time.sleep(0.05)
        exit_times.append(_real_time.perf_counter())
        return f"out-{prompt}"

    monkeypatch.setattr(HFLocalClient, "_generate_sync", fake_generate_sync, raising=True)

    async def go() -> None:
        client = HFLocalClient(model_id="fake/model", name="hf-test")
        await asyncio.gather(
            *(client.generate(f"p{i}", temperature=0.0, max_tokens=8, seed=0) for i in range(4))
        )

    _run(go())

    assert len(entry_times) == 4
    assert len(exit_times) == 4
    # Serial ordering: each subsequent entry is AFTER the previous exit.
    for i in range(1, 4):
        assert entry_times[i] >= exit_times[i - 1], (
            f"call {i} entered at {entry_times[i]:.4f} before previous exit "
            f"{exit_times[i - 1]:.4f} — generation lock is broken"
        )
