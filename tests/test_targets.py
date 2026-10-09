"""Tests for inragwetrust.data.targets (D31, D94, D95)."""

from inragwetrust.data.hotpotqa import Paragraph
from inragwetrust.data.pool import PoolQuestion
from inragwetrust.data.targets import (
    choose_candidates,
    group_key,
    is_valid_target,
    length_bucket,
)
from inragwetrust.textmatch import normalize


def pq(qid, answer, ans_text="Some text.", br_text="Other text.", question="Where?"):
    """A toy eligible question."""
    return PoolQuestion(
        id=qid, question=question, answer=answer, level="easy",
        answer_passage=Paragraph("A", (ans_text,)),
        bridge_passage=Paragraph("B", (br_text,)),
        distractors=(),
    )


def texts(q):
    return {
        "answer": normalize(q.answer),
        "question": normalize(q.question),
        "answer_passage": normalize(q.answer_passage.text),
        "bridge_passage": normalize(q.bridge_passage.text),
    }


def test_length_bucket():
    assert length_bucket("Paris") == 1
    assert length_bucket("one two three four five") == 4


def test_group_key():
    assert group_key("PERSON", "Ann Lee") == ("PERSON", 0)
    assert group_key("NONE", "Chief of Protocol") == ("NONE", 3)


def test_is_valid_target():
    q = pq("q1", "Paris", ans_text="Ann was born in Paris.", br_text="Ann visited Rome.")
    assert is_valid_target(q, normalize("London"), texts(q))
    assert not is_valid_target(q, normalize("Paris, Texas"), texts(q))  # contains a*
    assert not is_valid_target(q, normalize("Rome"), texts(q))  # in the bridge passage
    assert not is_valid_target(q, normalize("paris"), texts(q))  # same as a*


def test_choose_candidates_rules_and_reproducibility():
    pool = [pq("q1", "Paris"), pq("q2", "London"), pq("q3", "Rome"), pq("q4", "Paris, Texas")]
    fine = ["GPE"] * 4
    first = choose_candidates(pool, fine, seed=1)
    assert {t.answer for t in first["q1"]} == {"London", "Rome"}
    assert all(t.source_id != "q1" for t in first["q1"])
    assert first == choose_candidates(pool, fine, seed=1)


def test_labels_not_mixed_and_k_limit():
    pool = [pq("q1", "Paris"), pq("q2", "London"), pq("q3", "Rome"), pq("q5", "Ann Lee")]
    fine = ["GPE", "GPE", "GPE", "PERSON"]
    result = choose_candidates(pool, fine, seed=1, k=1)
    assert result["q5"] == []
    assert all(len(result[q]) == 1 for q in ("q1", "q2", "q3"))