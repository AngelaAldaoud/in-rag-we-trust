import pandas as pd
import pytest

from inragwetrust.analysis import confirmatory as cf


def test_controlled_parses_context_ids():
    df = pd.DataFrame({"context_id": ["clean-0.00", "swap-0.25", "closedbook", "bm25-swap"],
                       "rho": [0.0, 0.25, None, None],
                       "outcome": ["Correct", "Poisoned", "Abstain", "Correct"]})
    c = cf.controlled(df)
    assert c["context_id"].tolist() == ["clean-0.00", "swap-0.25"]
    assert c["poison_type"].tolist() == ["clean", "swap"] and c["rho"].tolist() == [0.0, 0.25]
    assert c["poisoned"].tolist() == [0, 1] and c["correct"].tolist() == [1, 0]


def test_controlled_rejects_inconsistent_rho():
    df = pd.DataFrame({"context_id": ["swap-0.25"], "rho": [0.5], "outcome": ["Correct"]})
    with pytest.raises(ValueError, match="disagrees"):
        cf.controlled(df)


def test_paired_stats_counts_and_direction():
    idx = pd.MultiIndex.from_tuples([("q1", 1), ("q1", 2), ("q2", 1), ("q2", 2)], names=["question_id", "k"])
    pairs = pd.DataFrame({"plain": [1, 1, 1, 0], "meta": [0, 0, 1, 0]}, index=idx)
    s = cf.paired_stats(pairs, "plain", "meta")
    assert (s["b"], s["c"], s["n"]) == (2, 0, 4) and s["estimate"] == -0.5
    assert s["ci_low"] <= -0.5 <= s["ci_high"]


def test_block_terms_selects_interaction_blocks():
    p = "C(prompt, Treatment('plain'))"
    names = (["Intercept", "rho", f"{p}[T.consistency]", "C(poison_type)[T.correct]"]
             + [f"rho:C(poison_type)[T.{t}]" for t in ("correct", "recency", "swap")]
             + [f"{p}[T.{q}]:C(poison_type)[T.{t}]" for q in ("consistency", "reliability")
                for t in ("correct", "recency", "swap")]
             + [f"rho:C(model)[T.{m}]" for m in ("mistral", "qwen")]
             + [f"{p}[T.{q}]:C(model)[T.{m}]" for q in ("consistency", "reliability") for m in ("mistral", "qwen")])
    blocks = cf.block_terms(names)
    assert {k: len(v) for k, v in blocks.items()} == {"rho:type": 3, "prompt:type": 6, "rho:model": 2, "prompt:model": 4}
    with pytest.raises(ValueError, match="prompt:model: expected 4 terms, found 0"):
        cf.block_terms(names[:-4])