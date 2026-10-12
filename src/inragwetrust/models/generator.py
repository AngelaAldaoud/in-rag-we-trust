"""vLLM generator for the evaluated models (C3, D101-D102, D119, D129-D130, D134-D135)."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass

from inragwetrust.conditions import MODELS
from inragwetrust.data.build import SEED
from inragwetrust.models.chat import REVISIONS

MAX_TOKENS = 512
MODES = {
    "greedy": {"temperature": 0.0, "n": 1},
    "samples": {"temperature": 0.7, "n": 5},
}


@dataclass(frozen=True)
class Request:
    condition_id: str
    question_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int
    finish_reason: str              # "stop" (model ended) or "length" (hit the MAX_TOKENS cap, D138)


def request_seed(condition_id: str, question_id: str, mode: str, seed: int = SEED) -> int:
    """A fixed 31-bit seed per request, independent of batch and order (D135)."""
    digest = hashlib.sha256(f"{seed}-{condition_id}-{question_id}-{mode}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


class Generator:
    """One evaluated model in vLLM; runs batches of token-ID prompts."""

    def __init__(self, model: str, engine=None, params_factory=None,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.90):
        self.model = model
        if engine is None:
            engine, params_factory = self._load(model, max_model_len, gpu_memory_utilization)
        self.engine, self.params_factory = engine, params_factory

    @staticmethod
    def _load(model, max_model_len, gpu_memory_utilization):
        from vllm import LLM, SamplingParams

        notebook_out, notebook_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__      # D119
        try:
            engine = LLM(model=MODELS[model], revision=REVISIONS[model],
                         tokenizer_revision=REVISIONS[model], dtype="bfloat16",
                         max_model_len=max_model_len,
                         gpu_memory_utilization=gpu_memory_utilization, seed=SEED)
        finally:
            sys.stdout, sys.stderr = notebook_out, notebook_err
        return engine, SamplingParams

    def params(self, mode: str, seed: int):
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        return self.params_factory(max_tokens=MAX_TOKENS, seed=seed, **MODES[mode])

    def generate(self, requests: list[Request], mode: str) -> list[list[Completion]]:
        """One list of completions per request, in request order (1 greedy or 5 samples)."""
        prompts = [{"prompt_token_ids": list(r.token_ids)} for r in requests]
        params = [self.params(mode, request_seed(r.condition_id, r.question_id, mode))
                  for r in requests]
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        return [[Completion(o.text, len(o.token_ids), o.finish_reason) for o in out.outputs]
                for out in outputs]