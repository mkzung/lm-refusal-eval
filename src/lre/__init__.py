"""lm-refusal-eval — reproducible refusal-rate harness for open-weight LLMs.

See the project README for design rationale, citations, and limitations.
"""

from __future__ import annotations

from lre.cache import ResponseCache
from lre.defense_in_depth import (
    PairedDefense,
    aggregate_paired_results,
    paired_label_responses,
    to_markdown_did,
)
from lre.judge import Judge, LLMJudge, RuleBasedJudge
from lre.provenance import Provenance, collect_provenance
from lre.report import from_json, scaling_table, to_json, to_markdown
from lre.runner import aggregate_results, ajudge_responses, judge_responses, run_eval
from lre.state import (
    EvalResult,
    Prompt,
    RawResponse,
    RefusalLabel,
    RunConfig,
)
from lre.stats import (
    Z_95,
    cohen_kappa,
    compute_proportion_diff_test,
    compute_wilson_ci,
)

__all__ = [
    "Z_95",
    "EvalResult",
    "Judge",
    "LLMJudge",
    "PairedDefense",
    "Prompt",
    "Provenance",
    "RawResponse",
    "RefusalLabel",
    "ResponseCache",
    "RuleBasedJudge",
    "RunConfig",
    "__version__",
    "aggregate_paired_results",
    "aggregate_results",
    "ajudge_responses",
    "cohen_kappa",
    "collect_provenance",
    "compute_proportion_diff_test",
    "compute_wilson_ci",
    "from_json",
    "judge_responses",
    "paired_label_responses",
    "run_eval",
    "scaling_table",
    "to_json",
    "to_markdown",
    "to_markdown_did",
]

__version__ = "0.1.0"
