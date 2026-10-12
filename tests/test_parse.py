"""Tests for final-answer extraction (D152) and the answer head (D156)."""

import pytest

from inragwetrust.evaluation.parse import clean, head, parse_answer


@pytest.mark.parametrize("text, answer, position", [
    ("The eponymous beverage consists of pineapple juice.\n\nAnswer: pineapple juice",          # pilot
     "pineapple juice", "line-start"),
    ("Reasoning: Passages [1], [2] ... no information.\n\nAnswer: unknown", "unknown", "line-start"),  # pilot
    ("...it is reasonable to infer that Bryant was associated with a film that won 10 Academy Awards. "
     'Answer: 10 Academy Awards (for the film "West Side Story")',                              # pilot
     '10 Academy Awards (for the film "West Side Story")', "mid-line"),
    ("Sachin H. Jain served as Chief Medical Information and Innovation Officer for Merck & Co., Inc. "
     "Answer: Merck.", "Merck", "mid-line"),                                                    # pilot
    ("...do not mention the specific northernmost and largest freguesia. Answer: unknown.",       # pilot
     "unknown", "mid-line"),
    ("**Answer:** Paris", "Paris", "line-start"),
    ("**Answer: Paris**", "Paris", "line-start"),
    ("**Answer**: Paris", "Paris", "line-start"),
    ("### Final Answer: Paris", "Paris", "line-start"),
    ("**Final Answer:** Paris", "Paris", "line-start"),
    ('Answer: "Magnetic Fields"', "Magnetic Fields", "line-start"),
    ("Answer: answer unknown", "unknown", "line-start"),
    ("**Answer:**\n\nParis\n", "Paris", "line-start"),
])
def test_answer_and_position(text, answer, position):
    parsed = parse_answer(text, "stop")
    assert (parsed.answer, parsed.position, parsed.truncated, parsed.cut) == (answer, position, False, False)


def test_the_last_answer_line_wins():
    assert parse_answer("Answer: Lyon\nWait, passage [2] disagrees.\nAnswer: Paris", "stop").answer == "Paris"


def test_plural_answers_is_not_an_answer_line():
    assert parse_answer("The answers: they vary across passages.", "stop").answer is None


def test_no_answer_line_model_stopped():                                                       # pilot (Llama)
    text = ("The question cannot be answered reliably because there is conflicting information about the "
            "number of Academy Awards won by the film in passages [1], [2], and [5].")
    parsed = parse_answer(text, "stop")
    assert (parsed.answer, parsed.position, parsed.truncated, parsed.cut) == (None, None, False, False)


def test_no_answer_line_truncated_loop():                                                      # pilot (Llama)
    text = ('a county in New Jersey is named "Pennington" is not a good fit however a county in New Jersey '
            'is named "Hunterdon" is not a good fit however a county in New Jersey is named "Pennington"')
    parsed = parse_answer(text, "length")
    assert (parsed.answer, parsed.truncated, parsed.cut) == (None, True, False)


def test_empty_answer_line_is_no_answer():
    parsed = parse_answer("Some reasoning.\nAnswer:", "stop")
    assert (parsed.answer, parsed.position) == (None, "line-start")


@pytest.mark.parametrize("text, answer", [
    ("Therefore ... California's 31st congressional district.\n\nAnswer: California's 31st", "California's 31st"),  # pilot
    ("Peter Helliar likely wrote the screenplay in 2009.\n\nAnswer: 2009.", "2009"),                               # pilot
])
def test_answer_at_the_cap_is_kept_and_flagged_cut(text, answer):
    parsed = parse_answer(text, "length")
    assert (parsed.answer, parsed.truncated, parsed.cut) == (answer, True, True)


def test_truncated_after_a_complete_answer_is_not_cut():
    parsed = parse_answer("Answer: Paris\nAnswer: Paris\nAnswer: Pa", "length")
    assert (parsed.answer, parsed.cut) == ("Pa", True)
    parsed = parse_answer("Answer: Paris\nThe passages agree on this point and", "length")
    assert (parsed.answer, parsed.cut) == ("Paris", False)


def test_clean_repeats_until_stable():
    assert clean(' **"Paris."** ') == "Paris"


@pytest.mark.parametrize("answer, expected", [
    ("New York (assuming Mercy College is located in New York, as it is ...)", "New York"),  # pilot
    ("Kiernan, 1892 (either James G. Kiernan or Sigmund Freud, depending ...)", "Kiernan, 1892"),  # pilot
    ("Paris [2]", "Paris"),
    ("A (b (c) d) e", "A e"),
    ("Olympic swimmer (in error due to the film's title, but he was", "Olympic swimmer"),
    ("(see passage 3)", "(see passage 3)"),
    ("Odyssey Through O or Magnetic Fields", "Odyssey Through O or Magnetic Fields"),       # pilot
])
def test_head_removes_bracketed_remarks(answer, expected):
    assert head(answer) == expected