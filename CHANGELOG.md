# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

Initial public release.

### Features

- **Adapters**: HuggingFace local (with chat-template toggle), OpenAI, Anthropic, and a deterministic `FakeModelClient` for tests and demos.
- **Judges**: precision-biased rule-based judge + LLM-judge protocol; defense-in-depth paired classifier with a Newcombe Method-10 CI on the Δ refusal rate.
- **Prompt suites**: `harmful_helpful`, `jailbreak_styles`, `refusal_calibration` shipped in `src/lre/data/`; suite hash baked into provenance.
- **Stats**: Wilson score CI on the headline rate, Cohen's κ for inter-judge agreement, two-proportion Newcombe interval, nearest-rank (Hyndman & Fan Type 1) p50/p99 latency.
- **Reproducibility**: byte-stable JSON, `SOURCE_DATE_EPOCH`-honouring provenance timestamps, deterministic seeded sampling, content-addressed response cache with atomic writes, `lre reproduce` to rebuild original invocations byte-for-byte.
- **CLI**: `lre run`, `lre demo`, `lre compare`, `lre kappa`, `lre did` (defense-in-depth), `lre cache info|clear|migrate`, `lre lint`, `lre reproduce`.
- **Validation**: hand-labeled set at `examples/data/validation_set.jsonl` (N=30) with `examples/validate_judge.py` reporting accuracy + κ. CI pins rule-judge κ ≥ 0.70 via `tests/test_validation_set.py`.
- **Quality bar**: pytest, ruff lint + format, mypy `--strict`. CI on Python 3.10 / 3.11 / 3.12.
