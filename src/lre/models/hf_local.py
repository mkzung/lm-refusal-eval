"""HuggingFace transformers adapter for local CPU/GPU inference.

The ``torch`` and ``transformers`` imports are intentionally lazy: importing
them at module import time would (a) double the harness startup cost on
machines that never touch the local path and (b) fail confusingly in
environments that only intend to use the API adapters. We pay the cost the
first time :meth:`HFLocalClient.generate` is invoked.

Determinism: we call ``transformers.set_seed`` before every generation,
which internally seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA)
in one call. With ``temperature=0`` and greedy decoding, output is
byte-identical across reruns on the same hardware. CUDA introduces some
non-determinism in matrix-multiply kernels, so for true bit-identity stick
to CPU inference or pin ``torch.use_deterministic_algorithms(True)``.

Effective name vs ``model_id``
------------------------------
``HFLocalClient.name`` is the identifier used downstream by the runner
and the response cache. To prevent silent collisions between two clients
that load the same weights but produce different completions (because
one wraps the prompt in a chat template and the other does not), the
default ``name`` is ``"<model_id>@chat"`` when ``use_chat_template=True``
and plain ``model_id`` otherwise. The user can still override ``name``
explicitly; the suffix only kicks in for the default.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from lre.state import RawResponse

logger = logging.getLogger(__name__)

_MISSING_DEP_MSG = (
    "HFLocalClient requires the optional 'hf' extra. Install with "
    "`pip install lm-refusal-eval[hf]` (pulls in torch + transformers)."
)


class HFLocalClient:
    """Local HuggingFace causal-LM adapter.

    Parameters
    ----------
    model_id:
        Either a HuggingFace Hub repo id (``"Qwen/Qwen2-0.5B-Instruct"``)
        or a local path containing ``config.json`` + weights.
    name:
        Public identifier used in :class:`RawResponse.model` and in the
        report tables. Defaults to ``model_id`` so scaling pivots based on
        substrings like ``"0.5b"`` / ``"1.5b"`` work out of the box.
    device:
        Either ``"cpu"``, ``"cuda"``, ``"auto"`` (default — picks ``cuda``
        if available), or any explicit ``torch.device``-compatible string.
    dtype:
        Optional torch dtype name (e.g. ``"float16"``, ``"bfloat16"``).
        Defaults to whatever the model config specifies on CPU and
        ``float16`` on CUDA.
    use_chat_template:
        Apply the tokenizer's ``apply_chat_template`` to wrap the
        prompt as a single-turn user message before generation. This
        is **required** for modern instruct models (Qwen2-Instruct,
        Llama-3-Instruct, Mistral-Instruct, etc.) — without it the
        model receives raw prompt text and tends to produce garbage
        completions. Defaults to ``True`` because the typical use case
        is instruct models; set ``False`` for base / pretraining-only
        checkpoints. Tokenizers without ``apply_chat_template`` fall
        back gracefully to the raw prompt.
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        device: str = "auto",
        dtype: str | None = None,
        use_chat_template: bool = True,
    ) -> None:
        self.model_id = model_id
        # Bake the chat-template flag into the default ``name`` so cache
        # entries cannot collide between a chat-templated and a
        # raw-prompt run of the same weights — those produce materially
        # different completions and must hash to different cache keys.
        if name is not None:
            self.name = name
        elif use_chat_template:
            self.name = f"{model_id}@chat"
        else:
            self.name = model_id
        self.device = device
        self.dtype = dtype
        self.use_chat_template = use_chat_template
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._resolved_device: str | None = None
        # Local-backend generation is single-stream by design. The lock
        # below serialises ``_generate_sync`` even when the runner fans
        # out multiple async tasks — without it, two concurrent threads
        # both call ``transformers.set_seed`` and ``torch.Generator``
        # state interleaves unpredictably, producing non-deterministic
        # output despite identical seeds. The GIL would force them to
        # serialise anyway; this lock keeps the determinism guarantee
        # explicit.
        self._generation_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the tokenizer and model on first use."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch  # intentional lazy import
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(_MISSING_DEP_MSG) from exc

        if self.device == "auto":
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self.device

        kwargs: dict[str, Any] = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.dtype)
        elif self._resolved_device == "cuda":
            kwargs["torch_dtype"] = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.to(self._resolved_device)
        self._model.eval()

    def _generate_sync(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        try:
            import torch
            from transformers import set_seed
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(_MISSING_DEP_MSG) from exc

        # `transformers.set_seed` seeds Python's `random`, NumPy, and
        # PyTorch (CPU + CUDA) — no need for a redundant `torch.manual_seed`
        # call.
        set_seed(seed)

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        # Apply the instruct-style chat template when configured AND the
        # loaded tokenizer actually supports it. Modern instruct models
        # require this wrapper — sending raw prompt text produces garbage
        # completions on Qwen2-Instruct, Llama-3-Instruct, etc. Base
        # models (no chat template) skip this branch entirely.
        if self.use_chat_template and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Some tokenizers expose ``apply_chat_template`` but
                # raise (e.g. no template registered, malformed Jinja).
                # Fall back to the raw prompt rather than blowing up an
                # entire eval; surface a warning so the operator can
                # decide whether to disable the flag explicitly.
                logger.warning(
                    "apply_chat_template raised on %s; falling back to raw prompt",
                    self.model_id,
                    exc_info=True,
                )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._resolved_device)

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Trim the prompt prefix; transformers returns input + completion.
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_len:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return str(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> RawResponse:
        start = time.perf_counter()
        # Hold the per-instance lock for the duration of the call so
        # ``transformers.set_seed`` + the model forward pass are atomic
        # relative to any other concurrent ``generate`` call. The lock
        # is awaited *before* ``asyncio.to_thread`` so the bookkeeping
        # cost is paid in the event loop, not the executor.
        async with self._generation_lock:
            text = await asyncio.to_thread(
                self._generate_sync,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
        elapsed = time.perf_counter() - start
        return RawResponse(
            prompt_id="",
            model=self.name,
            output=text,
            generation_seconds=elapsed,
            timestamp=int(time.time()),
            seed=seed,
        )
