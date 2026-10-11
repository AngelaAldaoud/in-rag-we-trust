"""Prompt texts (C2, D128, D131-D132)."""

import gzip
import json

import pytest

from inragwetrust.conditions import PROMPTS
from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.models.prompts import EXTRA, OUTPUT, messages, user_message

Q = "Which school?"
P = ["first passage", "second passage", "third passage"]


def test_prompts_differ_only_by_the_extra_sentence():
    plain = user_message(Q, P, "plain")
    for prompt in ("consistency", "reliability"):
        assert user_message(Q, P, prompt).replace(" " + EXTRA[prompt], "") == plain


@pytest.mark.parametrize("prompt", PROMPTS)
def test_structure(prompt):
    text = user_message(Q, P, prompt)
    assert text.endswith(OUTPUT) and '"Answer: unknown"' in text
    assert f"Question: {Q}" in text
    assert messages(Q, P, prompt) == [{"role": "user", "content": text}]


def test_passages_numbered_in_order_and_unchanged():
    text = user_message(Q, P, "plain")
    assert "[1] first passage\n\n[2] second passage\n\n[3] third passage" in text


def test_closed_book():
    text = user_message(Q, [], "plain")
    assert "Passages:" not in text and text.startswith("Answer the question.\n\n")
    with pytest.raises(ValueError):
        user_message(Q, [], "consistency")


def test_unknown_prompt():
    with pytest.raises(ValueError):
        user_message(Q, P, "fancy")


PATH = PROCESSED_DIR / "contexts.jsonl.gz"


@pytest.mark.skipif(not PATH.exists(), reason="contexts not built")
def test_all_real_prompts_render_and_passages_never_contain_answer_lines():
    with gzip.open(PATH, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    for r in rows:
        for cid, ctx in r["contexts"].items():
            texts = [p["text"] for p in ctx]
            assert all("Answer:" not in t for t in texts)
            for prompt in (["plain"] if cid == "closedbook" else PROMPTS):
                assert user_message(r["question"], texts, prompt)