"""Tests for inragwetrust.textmatch (decisions D46, D83, D85)."""

import pytest

from inragwetrust.textmatch import (
    contains_normalized,
    contains_span,
    find_spans,
    normalize,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("The United States!", "united states"),
        ("  Chief   of Protocol ", "chief of protocol"),
        ("An apple, a day", "apple day"),
        ("Theatre", "theatre"),  # "the" inside a word is kept
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_contains_normalized_whole_words_only():
    assert contains_normalized("the United States", "born in United States, 1950")
    assert not contains_normalized("art", "a party")


def test_contains_normalized_empty_answer():
    assert not contains_normalized("!!", "any text")


def test_find_spans_whole_words_any_case():
    assert find_spans("Paris", "Paris is not Parisian; paris again.") == [(0, 5), (23, 28)]


def test_find_spans_not_inside_longer_word():
    assert find_spans("US", "He moved to the USA.") == []


def test_find_spans_special_characters_taken_literally():
    assert contains_span("6.213 km", "The track is 6.213 km long.")
    assert not contains_span("6.213 km", "The track is 6x213 km long.")
    assert contains_span("C++", "She writes C++ code.")


def test_find_spans_empty_answer():
    assert find_spans("   ", "any text") == []