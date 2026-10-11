
"""Tests for inragwetrust.data.paraphrase_checks (D108)."""

from inragwetrust.data.paraphrase_checks import check_paraphrase

ORIG = "Brian Kirk (born 1968) is an Irish director. He directed the film My Boy Jack starring Daniel Radcliffe."
GOLD = "Daniel Radcliffe"
GOOD = "Born in 1968, Brian Kirk is an Irish director. He directed My Boy Jack, a film starring Daniel Radcliffe."


def test_good_paraphrase_passes():
    assert check_paraphrase(ORIG, GOOD, GOLD) == []


def test_new_name():
    para = GOOD.replace("Irish director", "Irish director from Dublin")
    assert "new_name" in check_paraphrase(ORIG, para, GOLD)


def test_missing_number():
    para = GOOD.replace("Born in 1968, ", "")
    assert "missing_number" in check_paraphrase(ORIG, para, GOLD)


def test_answer_missing():
    para = GOOD.replace("Daniel Radcliffe", "a young actor")
    failed = check_paraphrase(ORIG, para, GOLD)
    assert "answer_missing" in failed and "missing_name" in failed


def test_chatter_and_truncation():
    assert "chatter" in check_paraphrase(ORIG, "Here is the rewritten paragraph: " + GOOD, GOLD)
    assert "truncated" in check_paraphrase(ORIG, GOOD, GOLD, finish_reason="length")


def test_length_and_identical():
    assert "length" in check_paraphrase(ORIG, "Brian Kirk directed My Boy Jack with Daniel Radcliffe in 1968.", GOLD)
    assert "identical" in check_paraphrase(ORIG, ORIG, GOLD)


def test_capitals_inside_words_are_not_names():
    orig = "The album was sold on iTunes in 2010 by Apple."
    para = "In 2010, Apple sold the album on iTunes."
    assert check_paraphrase(orig, para, "Apple") == []


def test_elision_and_united_states_forms():
    orig = "It was recorded at the Château d'Hérouville by a U.S. band."
    para = "An American band recorded it at the Château d'Hérouville."
    assert check_paraphrase(orig, para, "Château d'Hérouville") == []