"""Generator logic with a fake engine (C3, D134-D135)."""

from types import SimpleNamespace

import pytest

from inragwetrust.models.generator import MAX_TOKENS, Generator, Request, request_seed


class FakeEngine:
    """Echoes each prompt's length; returns n outputs, the last one truncated."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, use_tqdm):
        self.calls.append((prompts, params))
        results = []
        for prompt, p in zip(prompts, params):
            n_in = len(prompt["prompt_token_ids"])
            outs = [SimpleNamespace(text=f"len={n_in} k={k}", token_ids=[0] * (k + 1),
                                    finish_reason="length" if k == p["n"] - 1 and p["n"] > 1 else "stop")
                    for k in range(p["n"])]
            results.append(SimpleNamespace(outputs=outs))
        return results


def make():
    engine = FakeEngine()
    return Generator("qwen", engine=engine, params_factory=dict), engine


REQS = [Request("c1", "q1", (1, 2, 3)), Request("c1", "q2", (1, 2, 3, 4, 5))]


def test_seeds_are_fixed_distinct_and_in_range():
    assert request_seed("c", "q", "samples") == request_seed("c", "q", "samples")
    seeds = {request_seed(c, q, m) for c in ("c1", "c2") for q in ("q1", "q2")
             for m in ("greedy", "samples")}
    assert len(seeds) == 8 and all(0 <= s < 2**31 for s in seeds)


def test_greedy_settings_and_order():
    gen, engine = make()
    out = gen.generate(REQS, "greedy")
    _, params = engine.calls[0]
    assert all(p["temperature"] == 0.0 and p["n"] == 1 and p["max_tokens"] == MAX_TOKENS
               for p in params)
    assert [o[0].text for o in out] == ["len=3 k=0", "len=5 k=0"]   # same order as requests


def test_samples_five_per_request_with_own_seed():
    gen, engine = make()
    out = gen.generate(REQS, "samples")
    _, params = engine.calls[0]
    assert all(p["temperature"] == 0.7 and p["n"] == 5 for p in params)
    assert [p["seed"] for p in params] == [request_seed("c1", q, "samples") for q in ("q1", "q2")]
    assert [len(o) for o in out] == [5, 5]
    assert out[0][4].finish_reason == "length" and out[0][4].n_tokens == 5


def test_unknown_mode():
    gen, _ = make()
    with pytest.raises(ValueError):
        gen.generate(REQS, "beam")