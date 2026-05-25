# lm-refusal-eval

**Reproducible refusal-rate evaluation harness for open-weight LLMs.**

[![CI](https://github.com/mkzung/lm-refusal-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/mkzung/lm-refusal-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Type-checked: mypy strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A small, well-tested harness for measuring how often a language model refuses adversarial prompts. Plug in a model (HF, OpenAI, Anthropic, or a deterministic synthetic adapter), run a fixed prompt suite, get JSON output that re-runs byte-identically given the same seed and `SOURCE_DATE_EPOCH`.

The motivating use case is the adversarial-robustness scaling-laws line of work — Howe, McKenzie, Hollinsworth, Zajac, Tseng, Tucker, Bacon, Gleave, *"Scaling Trends in Language Model Robustness"*, arXiv:2407.18213v5 (ICML 2025). Studies of that shape need a refusal harness that is (a) trivially extensible across model families and (b) reproducible enough that a number reported in a paper can be re-derived months later from the same prompt suite and the same seed.

## Quick start

```bash
pip install -e ".[dev]"
lre demo --sample 5
```

`lre demo` exercises every code path — runner, judge, aggregator, scaling-table renderer — against the deterministic `FakeModelClient`, with no network or GPU required. CI runs the same flow on every commit.

For a real model:

```bash
# Local HuggingFace
lre run \
    --adapter hf --model qwen-0.5b \
    --model-id Qwen/Qwen2-0.5B-Instruct \
    --suite harmful_helpful --out results.json

# OpenAI
lre run \
    --adapter openai --model gpt-4o-mini \
    --suite harmful_helpful --out results.json

# Anthropic
lre run \
    --adapter anthropic --model claude-3-haiku-20240307 \
    --suite harmful_helpful --out results.json
```

## Judge validation

The rule judge is documented as **precision-biased** — false negatives are more common than false positives. `examples/data/validation_set.jsonl` is a hand-labeled set of 30 prompt-response pairs spanning the easy and adversarial surface (clear refusals, clear compliance, refusal-in-quote, mid-text refusal, `"I will not lie"` false-positive bait, prompt-level verdict-shaped phrases). Run:

```bash
python examples/validate_judge.py
```

Current measurement:

| Metric | Value |
|---|---|
| Accuracy | 0.900 (27/30) |
| Cohen's κ | 0.783 — substantial agreement (Landis & Koch 1977) |
| False positives | 0/18 — judge is precision-biased as advertised |
| False negatives | 3/12 — refusals using phrasings outside the trigger set |

`tests/test_validation_set.py` pins κ ≥ 0.70 in CI, so any judge regression that hurts agreement fails the build.

## What's measured

For each `(model, suite)` pair, `lre run` produces:

- **Per-prompt refusal labels** (rule judge or LLM judge) with the matched pattern(s) for auditability.
- **Aggregate refusal rate** with a 95% **Wilson score** confidence interval — at N=10..15 the half-width on a 0.5 rate is roughly ±0.25, so headline rates without bounds are misleading.
- **Per-category breakdown** (harmful / helpful / borderline) so over-refusal and under-refusal surface separately.
- **Paired-defense (defense-in-depth) joint refusal rate** for two-judge layered pipelines, with a corrected **Newcombe Method-10** confidence interval on the Δ rate vs. a single judge.
- **Provenance block** — `schema_version`, `lre_version`, Python version, platform, hashed hostname, git SHA + dirty flag, seed, ISO-8601 UTC timestamp, and the full set of CLI inputs needed by `lre reproduce` to rebuild the original invocation byte-for-byte.

## Reproducibility

| Property | Mechanism |
|---|---|
| Deterministic synthetic client | SHA-256-based per-prompt hash; no Python `hash()` |
| Deterministic local generation | `transformers.set_seed` per call (Python `random`, NumPy, PyTorch CPU + CUDA) |
| Byte-stable JSON | `sort_keys=True`, `indent=2`, `allow_nan=False`, trailing newline |
| Byte-stable timestamps | `SOURCE_DATE_EPOCH=<unix-epoch>` honoured |
| Content-addressed cache | `--cache .lre-cache/`; keys on `SHA256(model | prompt | seed | temp | max_tokens)`; shard layout survives across reruns |
| Result provenance | `schema_version` = `"1.0"`; external tooling dispatches without parsing the rest of the file |
| Deterministic sampling | `--sample N --seed K` uses `random.Random(K).sample(...)`; tag `[sampled N/M, seed=K]` propagates into the result row |
| Reproduce | `lre reproduce results.json` prints the equivalent `lre run` invocation; `--exec` re-runs it |

## Limitations

- The rule judge is precision-biased by design — see the validation set above. For high-recall evaluation, use `--judge llm` with a paired defense-in-depth setup.
- Refusal-rate measurements are sensitive to prompt-suite framing. A model that scores low on one suite may score high on another; cross-suite comparisons require the same suite hash. The Provenance block records `suite_hash` for exactly this reason.
- Local HF generation on small models (≤7B) is deterministic at temperature 0 with `set_seed`; larger models and 4-bit/8-bit quantisation introduce non-determinism that the harness does not paper over.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
mypy --strict src/
pytest tests/ -v --cov=src/lre --cov-report=term-missing
```

CI runs the same on Python 3.10 / 3.11 / 3.12 on `ubuntu-latest`.

## Citation

If you use this harness, please cite both the scaling-laws paper that motivates it and this repository:

```bibtex
@article{howe2024scaling,
  title   = {Scaling Trends in Language Model Robustness},
  author  = {Howe, Nikolaus and McKenzie, Ian and Hollinsworth, Oskar
             and Zajac, Micha{\l} and Tseng, Tom and Tucker, Aaron
             and Bacon, Pierre-Luc and Gleave, Adam},
  journal = {arXiv preprint arXiv:2407.18213},
  year    = {2024}
}

@software{lm_refusal_eval,
  author = {Gorbuk, Max},
  title  = {lm-refusal-eval: Reproducible refusal-rate evaluation harness for open-weight LLMs},
  url    = {https://github.com/mkzung/lm-refusal-eval},
  year   = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
