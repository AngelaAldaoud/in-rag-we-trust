"""Tests for inragwetrust.data.entity_types (D87, D91)."""

import pytest
import spacy
from spacy.tokens import Span

from inragwetrust.data.entity_types import (
    NO_ENTITY,
    coarse_type,
    label_in_doc,
    tag_answers,
)


@pytest.fixture
def doc():
    """A hand-labelled document, independent of the statistical model."""
    nlp = spacy.blank("en")
    d = nlp("Ann Lee was born in Paris in 1950.")
    # tokens: Ann(0) Lee(1) was(2) born(3) in(4) Paris(5) in(6) 1950(7) .(8)
    d.ents = [
        Span(d, 0, 2, label="PERSON"),
        Span(d, 5, 6, label="GPE"),
        Span(d, 7, 8, label="DATE"),
    ]
    return d


@pytest.mark.parametrize(
    "answer, expected",
    [("Paris", "GPE"), ("Ann Lee", "PERSON"), ("born", NO_ENTITY), ("London", NO_ENTITY)],
)
def test_label_in_doc(doc, answer, expected):
    assert label_in_doc(answer, doc) == expected


def test_entity_must_cover_half_of_answer(doc):
    # "Ann Lee" covers 7 of the 16 characters of "Ann Lee was born": less than half.
    assert label_in_doc("Ann Lee was born", doc) == NO_ENTITY


def test_coarse_type_mapping():
    assert coarse_type("GPE") == "place"
    assert coarse_type("CARDINAL") == "number"
    assert coarse_type("NORP") == "other"
    assert coarse_type(NO_ENTITY) == "other"


def test_tag_answers_with_real_model():
    text = "As an adult, she was named United States ambassador to Ghana."
    assert tag_answers([("Ghana", text)]) == ["GPE"]
    