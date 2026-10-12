"""Tests for the metric definitions (D153, D170-D175) on hand-made responses."""

import pytest

from inragwetrust.evaluation import metrics as m


def row(outcome, answer, qid="q1", cid="c1", run="poisoned", hall=None, truncated=False, tokens=10, sample=0):
    return {"outcome": outcome, "answer": answer, "question_id": qid, "condition_id": cid, "run": run,
            "hall": hall, "truncated": truncated, "n_tokens": tokens, "sample": sample}


GOLDS = {"q1": "Paris", "q2": "Paris", "q3": "Paris", "q4": "Paris", "q5": "Avatar (2009 film)"}


def test_rate_carries_wilson_interval():
    r = m.rate(5, 10)
    assert (r.k, r.n, r.value) == (5, 10, 0.5) and r.lo == pytest.approx(0.2366, abs=1e-3)


@pytest.mark.parametrize("answer, gold, em, f1", [
    ("Paris", "Paris", 1.0, 1.0),
    ("the city of Paris", "Paris", 0.0, 0.5),
    ("Avatar", "Avatar (2009 film)", 1.0, 1.0),
    ("Lyon", "Paris", 0.0, 0.0),
])
def test_em_f1(answer, gold, em, f1):
    assert m.em_f1(answer, gold) == pytest.approx((em, f1))


def test_abstentions_and_missing_answers_score_zero():
    assert m.response_em_f1(row("Abstain", "unknown"), "Paris") == (0.0, 0.0)
    assert m.response_em_f1(row("NoAnswer", None), "Paris") == (0.0, 0.0)


def test_sc_value_maps_abstentions_and_missing_answers():
    assert m.sc_value(row("Abstain", "not mentioned in the passages")) == "abstain"
    assert m.sc_value(row("NoAnswer", None)) == "no answer"
    assert m.sc_value(row("OtherWrong", "The Marseille (port city)")) == "marseille"


def test_upper_outcome_only_relabels_other_wrong():
    assert m.upper_outcome(row("OtherWrong", "IWC"), "G") == "Correct"
    assert m.upper_outcome(row("OtherWrong", "x"), "T") == "Poisoned"
    assert m.upper_outcome(row("OtherWrong", "x"), "N") == "OtherWrong"
    assert m.upper_outcome(row("OtherWrong", "x"), None) == "OtherWrong"
    assert m.upper_outcome(row("Poisoned", "Lyon"), "G") == "Poisoned"


def test_greedy_metrics_on_a_known_group():
    rows = [row("Correct", "Paris", "q1", "c1"),
            row("Poisoned", "Lyon", "q2", "c2", truncated=True, tokens=30),
            row("OtherWrong", "Marseille", "q3", "c3", hall=True),
            row("NoAnswer", None, "q4", "c4"),
            row("Abstain", "unknown", "q5", "c5", run="closed_book")]
    flags = {("c1", "q1"): "Y", ("c2", "q2"): "N", ("c3", "q3"): None, ("c4", "q4"): "Y", ("c5", "q5"): "Y"}
    verdicts = {("c3", "q3"): "G"}
    out = m.greedy_metrics(rows, GOLDS, flags, verdicts)
    assert [out[k].k for k in ("acc", "asr", "other_wrong", "abstain", "no_answer", "no_usable")] == [1, 1, 1, 1, 1, 2]
    assert out["acc"].n == 5
    assert (out["hall"].k, out["hall"].n, out["hall_given_wrong"].value) == (1, 5, 1.0)
    assert (out["flag"].k, out["flag"].n, out["flag_missing"]) == (2, 3, 1)       # closed-book excluded
    assert (out["acc_plus"].k, out["asr_plus"].k) == (2, 1)
    assert out["em"] == pytest.approx(0.2) and out["truncated"].k == 1 and out["mean_tokens"] == pytest.approx(14)


def test_greedy_metrics_refuses_an_empty_group():
    with pytest.raises(ValueError):
        m.greedy_metrics([], GOLDS, {}, {})


def test_sc_on_answers_and_on_outcomes():
    mixed = [row("Correct", "Paris", sample=0), row("Correct", "paris.", sample=1), row("Poisoned", "Lyon", sample=2),
             row("Abstain", "unknown", sample=3), row("NoAnswer", None, sample=4)]
    refusals = [row("Abstain", text, qid="q2", sample=i) for i, text in enumerate(
        ["unknown", "not mentioned", "cannot be determined", "unknown.", "not provided"])]
    out = m.sc_metrics(mixed + refusals)
    assert out["questions"] == 2
    assert out["sc"] == pytest.approx((0.4 + 1.0) / 2) and out["sc_outcome"] == pytest.approx((0.4 + 1.0) / 2)