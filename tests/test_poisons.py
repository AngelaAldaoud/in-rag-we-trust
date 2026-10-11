"""Tests for inragwetrust.poisoning.poisons (D112)."""

import random

import pytest

from inragwetrust.poisoning.poisons import Swap, swap_answer


def test_replaces_every_occurrence_any_case():
    text = "Paris is big. I love paris."
    assert swap_answer(text, "Paris", "Rome") == "Rome is big. I love Rome."


def test_not_inside_longer_words():
    assert swap_answer("A Parisian lives in Paris.", "Paris", "Rome") == "A Parisian lives in Rome."


def test_capitalises_target_at_sentence_start():
    text = "The British military arrested him."
    assert swap_answer(text, "the British military", "French and English") == \
        "French and English arrested him."


def test_missing_answer_raises():
    with pytest.raises(ValueError):
        swap_answer("Nothing here.", "Paris", "Rome")


def test_swap_adds_nothing():
    text = "Ann Lee was born in Paris."
    assert Swap().apply(text, "Paris", "Rome", random.Random(1)) == "Ann Lee was born in Rome."



from inragwetrust.poisoning.poisons import POISON_TYPES, Authority, Correct, Recency
from inragwetrust.textmatch import contains_span

TEXT = "Ann Lee was born in Paris."


def test_framed_types_add_exactly_one_template_sentence():
    swapped = "Ann Lee was born in Rome."
    for cls in (Correct, Authority, Recency):
        out = cls().apply(TEXT, "Paris", "Rome", random.Random(5))
        assert out.startswith(swapped + " ")
        assert out[len(swapped) + 1:] in cls.templates


def test_same_seed_same_poison():
    for p in POISON_TYPES:
        a = p.apply(TEXT, "Paris", "Rome", random.Random(9))
        b = p.apply(TEXT, "Paris", "Rome", random.Random(9))
        assert a == b


def test_templates_are_neutral_and_distinct():
    all_templates = [t for p in POISON_TYPES for t in p.templates]
    assert len(all_templates) == len(set(all_templates)) == 15
    assert not any(ch.isdigit() for t in all_templates for ch in t)
    assert not any(contains_span("Rome", t) or contains_span("Paris", t) for t in all_templates)


def test_four_types_in_fixed_order():
    assert [p.name for p in POISON_TYPES] == ["swap", "correct", "authority", "recency"]



from inragwetrust.poisoning.build import answer_shape, choose_target


def test_answer_shape():
    assert answer_shape("April 20, 1946") == (True, 2, False)
    assert answer_shape("1902") == (False, 1, False)
    assert answer_shape("between 1932 and 1934") == (False, 2, True)
    assert answer_shape("the mayor") == (False, 0, False)


def test_choose_target_matches_shape_for_dates():
    q = {"answer_type": "date", "gold_answer": "April 20, 1946",
         "target_candidates": [{"answer": "1902"}, {"answer": "March 3, 1975"}]}
    assert choose_target(q) == ("March 3, 1975", True)


def test_choose_target_unchanged_for_people():
    q = {"answer_type": "person", "gold_answer": "Daniel Radcliffe",
         "target_candidates": [{"answer": "Gareth"}, {"answer": "Bill Paxton"}]}
    assert choose_target(q) == ("Gareth", True)


