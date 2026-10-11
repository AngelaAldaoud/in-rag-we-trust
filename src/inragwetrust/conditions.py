"""The 192 experimental conditions of the design (C1, A3 design table, D125)."""

from __future__ import annotations

from dataclasses import dataclass

from inragwetrust.retrieval.contexts import RHOS, TYPE_NAMES

MODELS = {                                    # short key -> HuggingFace model ID
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
PROMPTS = ("plain", "consistency", "reliability")
RUNS = ("closed_book", "baseline", "poisoned", "retrieval")


@dataclass(frozen=True)
class Condition:
    condition_id: str                         # e.g. "llama__reliability__authority-0.50"
    run: str                                  # one of RUNS
    model: str                                # key of MODELS
    prompt: str                               # one of PROMPTS
    poison_type: str | None                   # None for closed-book and the baseline
    rho: float | None                         # None for closed-book and retrieval
    context_id: str                           # which of the 22 contexts per question


def context_id(run: str, poison_type: str | None = None, rho: float | None = None) -> str:
    """Name of the context a condition reads (independent of model and prompt)."""
    if run == "closed_book":
        return "closedbook"
    if run == "baseline":
        return "clean-0.00"
    if run == "poisoned":
        return f"{poison_type}-{rho:.2f}"
    if run == "retrieval":
        return f"bm25-{poison_type}"
    raise ValueError(f"unknown run: {run}")


def _condition(run, model, prompt, poison_type=None, rho=None) -> Condition:
    ctx = context_id(run, poison_type, rho)
    return Condition(f"{model}__{prompt}__{ctx}", run, model, prompt, poison_type, rho, ctx)


def condition_table() -> list[Condition]:
    """All conditions, model by model, so each model is loaded only once (A3)."""
    rows = []
    for model in MODELS:
        rows.append(_condition("closed_book", model, "plain"))
        for prompt in PROMPTS:
            rows.append(_condition("baseline", model, prompt, None, 0.0))
            for ptype in TYPE_NAMES:
                for rho in RHOS[1:]:
                    rows.append(_condition("poisoned", model, prompt, ptype, rho))
            for ptype in TYPE_NAMES:
                rows.append(_condition("retrieval", model, prompt, ptype))
    return rows


def context_ids() -> list[str]:
    """The distinct context IDs, in table order (22 per question)."""
    return list(dict.fromkeys(c.context_id for c in condition_table()))