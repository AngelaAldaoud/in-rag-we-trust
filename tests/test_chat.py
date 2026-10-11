"""Chat rendering logic (C2, D129-D130), tested with a fake tokenizer."""

import re

import pytest

from inragwetrust.conditions import MODELS
from inragwetrust.models.chat import REVISIONS, render


class FakeTokenizer:
    """Mistral-like: the template writes '<s>', which is token 1."""

    bos_token_id = 1
    prefix = "<s>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"{self.prefix}[INST] {messages[0]['content']}[/INST]"

    def __call__(self, text, add_special_tokens):
        ids = [1] if add_special_tokens else []
        while text.startswith("<s>"):
            ids.append(1)
            text = text[3:]
        return {"input_ids": ids + [ord(c) for c in text]}


def msg(content):
    return [{"role": "user", "content": content}]


def test_revisions_are_pinned_full_commits():
    assert set(REVISIONS) == set(MODELS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in REVISIONS.values())


def test_exactly_one_start_token():
    p = render(FakeTokenizer(), msg("hello"))
    assert p.token_ids[0] == 1 and p.token_ids[1] != 1
    assert p.text == "<s>[INST] hello[/INST]" and p.n_tokens == len(p.token_ids)


def test_hash_is_stable_and_sensitive():
    tok = FakeTokenizer()
    assert render(tok, msg("a")).prompt_hash == render(tok, msg("a")).prompt_hash
    assert render(tok, msg("a")).prompt_hash != render(tok, msg("b")).prompt_hash


def test_duplicated_start_token_is_refused():
    class Broken(FakeTokenizer):
        prefix = "<s><s>"

    with pytest.raises(ValueError):
        render(Broken(), msg("hello"))