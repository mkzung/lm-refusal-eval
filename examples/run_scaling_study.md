# Reference scaling study (synthetic)

> **Note.** This file is a worked example using `FakeModelClient` — the
> refusal numbers below are deterministic output from the synthetic adapter,
> not measurements on any real model. Treat it as a documentation aid for
> what `lre` output looks like when wired to a series of model sizes.

## Reproducing this report

The snippet below demonstrates the v0.2+ surface end-to-end:

* `aggregate_results(prompts=...)` populates the per-category breakdown
  (harmful / helpful / borderline) so over-refusal and under-refusal are
  visible at a glance.
* `refusal_rate_ci_low` / `refusal_rate_ci_high` carry a 95% **Wilson
  score** confidence interval on each headline rate — at N=10..15 the
  half-width on a 0.5 rate is roughly ±0.25, so a "look at the dot plot
  and squint" reading of small-N rate differences is misleading without
  the bounds.
* `aggregate_results` accepts the prompt list so all-errored categories
  surface as `None` rather than silently disappearing from the report.

```python
import asyncio

from lre.synthetic import FakeModelClient
from lre.runner import run_eval, ajudge_responses, aggregate_results
from lre.report import to_markdown, scaling_table
from lre.prompts import load_suite
from lre.state import RunConfig

async def study() -> None:
    results = []
    for size, name in [
        (0.5, "qwen-mock-0.5b"),
        (1.5, "qwen-mock-1.5b"),
        (7.0, "qwen-mock-7b"),
    ]:
        # Synthetic refusal rate scales weakly with size, mirroring the
        # qualitative trend reported by Howe, McKenzie, Hollinsworth,
        # Zajac, Tseng, Tucker, Bacon, Gleave — *Effects of Scale on
        # Language Model Robustness* (arXiv:2407.18213, 2024) — for
        # several model families. Replace FakeModelClient with
        # HFLocalClient or an API adapter for real measurements.
        refusal_rate = min(0.3 + size / 15, 0.95)
        client = FakeModelClient(name=name, refusal_rate=refusal_rate, seed=42)
        for suite in ["harmful_helpful", "jailbreak_styles", "refusal_calibration"]:
            cfg = RunConfig(model=name, suites=[suite], seed=42)
            prompts = load_suite(suite)
            responses = await run_eval(client, suite, cfg)
            labels = await ajudge_responses(responses)
            results.append(
                aggregate_results(
                    model=name,
                    suite=suite,
                    responses=responses,
                    labels=labels,
                    # Passing `prompts=` enables the per-category breakdown.
                    prompts=prompts,
                )
            )
    print(to_markdown(results, title="Synthetic scaling study"))
    print()
    print(scaling_table(results))

asyncio.run(study())
```

## Discussion

A real run of this script against open-weight checkpoints (e.g.
`Qwen/Qwen2-0.5B-Instruct`, `Qwen/Qwen2-1.5B-Instruct`, `Qwen/Qwen2-7B-Instruct`)
would let you reproduce a refusal-rate-vs-size curve in roughly an hour on
a single A100, or a few hours on CPU. The harness keeps results in a
structured `EvalResult` row — headline rate, Wilson 95% CI bounds, per-
category breakdown, p50 / p99 latency — so the downstream analysis
(fitting a power law, plotting confidence-banded curves, computing
inter-judge agreement via `lre kappa`) can live in a separate notebook
without touching the harness.

A few notes on interpreting the output:

* Take the 95% Wilson CI seriously at small N. With 15 prompts and an
  observed rate of 0.5, the interval is roughly `[0.27, 0.73]` — a single
  refusal flip moves the headline rate by ~0.07 but the CI barely
  budges. Bigger suites narrow the bounds at √N.
* The per-category sub-table is the cheap way to spot over-refusal:
  if `helpful` refusal rate climbs alongside `harmful`, the model is
  paying for safety with capability. `harmful_helpful` includes both
  classes by construction; `jailbreak_styles` is all-`harmful` so its
  `helpful` column will render as `—`.
* Use `lre compare results_a.json results_b.json` to formalise a
  size-vs-size delta with a two-proportion z-test and a 95% CI on Δ.
  At small per-suite N you often find that visually-striking gaps fail
  to reach α = 0.05 — exactly the situation Wilson bounds are there for.

The intent is to make refusal-rate measurement *boring*: the scientifically
interesting part of a paper happens *after* the EvalResult JSON is on disk,
not inside the harness itself. Keeping the harness narrow makes it easier
to plug into FAR.AI-style scaling-laws workflows alongside other axes of
variation (attack type, training-data mixture, decoding strategy, etc.).
