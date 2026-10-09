"""Tests for inragwetrust.data.hotpotqa."""

import pytest

from inragwetrust.data.hotpotqa import (
    DEV_FILENAME,
    RAW_DIR,
    HotpotExample,
    Paragraph,
    load_dev,
    sha256_of,
)

# A tiny hand-made row with the same structure as a real HotpotQA row.
ROW = {
    "id": "q1",
    "question": "Where was the director of Film X born?",
    "answer": "Paris",
    "type": "bridge",
    "level": "easy",
    "supporting_facts": {
        "title": ["Film X", "Ann Lee", "Ann Lee"],
        "sent_id": [0, 0, 1],
    },
    "context": {
        "title": ["Ann Lee", "Other", "Film X"],
        "sentences": [
            ["Ann Lee is a director.", " She was born in Paris."],
            ["Unrelated text."],
            ["Film X is a film by Ann Lee."],
        ],
    },
}


def test_paragraph_text_joins_without_extra_spaces():
    p = Paragraph("T", ("First.", " Second."))
    assert p.text == "First. Second."


def test_from_row_fields():
    ex = HotpotExample.from_row(ROW)
    assert ex.id == "q1"
    assert ex.answer == "Paris"
    assert ex.type == "bridge"
    assert len(ex.paragraphs) == 3


def test_supporting_titles_deduplicated_in_order():
    ex = HotpotExample.from_row(ROW)
    assert ex.supporting_titles == ("Film X", "Ann Lee")


def test_supporting_paragraphs_in_context_order():
    ex = HotpotExample.from_row(ROW)
    assert [p.title for p in ex.supporting_paragraphs()] == ["Ann Lee", "Film X"]


def test_examples_are_frozen():
    ex = HotpotExample.from_row(ROW)
    with pytest.raises(AttributeError):
        ex.answer = "London"


def test_sha256_of(tmp_path):
    f = tmp_path / "abc.txt"
    f.write_bytes(b"abc")
    assert sha256_of(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


needs_raw = pytest.mark.skipif(
    not (RAW_DIR / DEV_FILENAME).exists(),
    reason="raw HotpotQA file not downloaded",
)


@needs_raw
def test_load_dev_real_file():
    ex = load_dev()
    assert len(ex) == 7405
    corliss = next(x for x in ex if x.id == "5a8c7595554299585d9e36b6")
    assert corliss.answer == "Chief of Protocol"
    assert len(corliss.supporting_paragraphs()) == 2