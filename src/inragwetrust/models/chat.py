"""Chat-template rendering per model (C2, D128-D130)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from inragwetrust.conditions import MODELS

REVISIONS = {                     # pinned HuggingFace commits (D130)
    "qwen": "a09a35458c702b33eeacc393d103063234e8bc28",
    "llama": "0e9e39f249a16976918f6564b8830bc894c89659",
    "mistral": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}


@dataclass(frozen=True)
class RenderedPrompt:
    text: str                     # what the model reads, for display and hashing
    token_ids: tuple[int, ...]    # what is sent to vLLM (D129)
    prompt_hash: str              # SHA-256 of the text

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


def load_tokenizer(model: str):
    """The model's tokenizer at its pinned revision (needs transformers: Colab only, D90)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[model], revision=REVISIONS[model])


def render(tokenizer, messages: list[dict]) -> RenderedPrompt:
    """Apply the chat template, then tokenize once without adding special tokens."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is not None and ids[:2] == [bos, bos]:
        raise ValueError("prompt starts with a duplicated start token")
    return RenderedPrompt(text, tuple(ids), hashlib.sha256(text.encode("utf-8")).hexdigest())