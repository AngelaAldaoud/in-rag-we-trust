"""Tests for inragwetrust.data.pool (filters D29, D83, D84)."""

from inragwetrust.data.hotpotqa import HotpotExample
from inragwetrust.data.pool import build_pool, is_yes_no, split_passages


def make(qid, answer, qtype, first_text, second_text):
    """A toy question with supporting paragraphs 'First' and 'Second' and one distractor."""
    row = {
        "id": qid,
        "question": "Q?",
        "answer": answer,
        "type": qtype,
        "level": "easy",
        "supporting_facts": {"title": ["First", "Second"], "sent_id": [0, 0]},
        "context": {
            "title": ["Noise", "First", "Second"],
            "sentences": [["Noise text."], [first_text], [second_text]],
        },
    }
    return HotpotExample.from_row(row)


GOOD = make("good", "Paris", "bridge", "Ann Lee directed Film X.", "Ann Lee was born in Paris.")
YESNO = make("yesno", "yes", "comparison", "A is old.", "B is old.")
BOTH = make("both", "Paris", "bridge", "Paris is big.", "Ann Lee was born in Paris.")
LEAK = make("leak", "U.S. Army", "bridge", "He joined the U.S. Army.", "He later left the US Army.")
COMPARISON = make("comp", "Paris", "comparison", "Paris is old.", "Rome is old.")


def test_is_yes_no():
    assert is_yes_no("yes") and is_yes_no(" No ")
    assert not is_yes_no("Paris")


def test_roles_of_a_good_question():
    answer_passage, bridge_passage = split_passages(GOOD)
    assert answer_passage.title == "Second"
    assert bridge_passage.title == "First"


def test_answer_in_both_paragraphs_is_rejected():
    assert split_passages(BOTH) is None


def test_normalised_leak_in_bridge_is_rejected():
    # "U.S. Army" is not a span of "US Army", but both normalise to "us army".
    assert split_passages(LEAK) is None


def test_funnel_and_pool():
    pool, funnel = build_pool([GOOD, YESNO, BOTH, LEAK, COMPARISON])
    assert funnel == {"all": 5, "not_yes_no": 4, "answer_in_exactly_one": 2, "bridge_type": 1}
    assert [q.id for q in pool] == ["good"]
    assert [p.title for p in pool[0].distractors] == ["Noise"]