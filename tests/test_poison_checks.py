"""Tests for inragwetrust.poisoning.checks (D116)."""

from inragwetrust.poisoning.checks import check_poison
from inragwetrust.poisoning.poisons import Correct

CLEAN = "Ann Lee was born in Paris."
SWAPPED = "Ann Lee was born in Rome."

from inragwetrust.poisoning.checks import leak_tokens


def test_word_shared_with_target_is_not_a_leak():
    assert leak_tokens("Love Streams", "Rainbow Streams") == (set(), None)
    assert leak_tokens("February 9, 1994", "February 9, 1984") == ({"1994"}, None)

def test_good_swap():
    res = check_poison(CLEAN, SWAPPED, SWAPPED, "Paris", "Rome", "swap", ())
    assert res["blocking"] == [] and res["partial"] == []


def test_gold_still_present():
    poisoned = "Ann Lee was born in Rome, not Paris."
    res = check_poison(CLEAN, poisoned, poisoned, "Paris", "Rome", "swap", ())
    assert "gold_present" in res["blocking"]


def test_framing_must_be_swap_plus_own_template():
    good = SWAPPED + " " + Correct.templates[0]
    assert check_poison(CLEAN, good, SWAPPED, "Paris", "Rome", "correct", Correct.templates)["blocking"] == []
    bad = SWAPPED + " This is definitely true."
    assert "framing" in check_poison(CLEAN, bad, SWAPPED, "Paris", "Rome", "correct", Correct.templates)["blocking"]


def test_number_leak_but_not_shared_numbers():
    clean = "A 1946 film released on April 20, 1946."
    leaky = "A 1946 film released on September 25, 2017."
    res = check_poison(clean, leaky, leaky, "April 20, 1946", "September 25, 2017", "swap", ())
    assert "number_leak" in res["partial"]
    shared = "A film released on September 20, 2017."
    res = check_poison(clean, shared, shared, "April 20, 1946", "September 20, 2017", "swap", ())
    assert "number_leak" not in res["partial"]


def test_word_leak():
    poisoned = "It starred Gareth. Radcliffe later left."
    res = check_poison("x", poisoned, poisoned, "Daniel Radcliffe", "Gareth", "swap", ())
    assert "word_leak" in res["partial"]


def test_generic_last_word_is_not_a_leak():
    poisoned = "It was arrested by French and English forces; military service followed."
    res = check_poison("x", poisoned, poisoned, "the British military", "French and English", "swap", ())
    assert "word_leak" not in res["partial"]


from inragwetrust.poisoning.build import select_target


def test_select_target_prefers_safe_generated():
    q = {"id": "q1", "answer_type": "person", "gold_answer": "Daniel Radcliffe",
         "target_candidates": [{"answer": "Gareth"}]}
    assert select_target(q, {"q1": {"target": "Rupert Grint", "safe": True}}) == ("Rupert Grint", "generated")
    assert select_target(q, {"q1": {"target": "Daniel", "safe": False}}) == ("Gareth", "fallback_candidate")
    assert select_target(q, {}) == ("Gareth", "fallback_candidate")
    