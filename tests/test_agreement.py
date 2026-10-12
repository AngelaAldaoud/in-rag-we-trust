"""Tests for the agreement statistics (D167)."""

import math

import pytest

from inragwetrust.evaluation import agreement as ag


def test_kappa_known_value_and_perfect_agreement():
    assert ag.cohen_kappa(list("YYNN"), list("YNNN")) == pytest.approx(0.5)
    assert ag.cohen_kappa(list("YNYN"), list("YNYN")) == pytest.approx(1.0)


def test_kappa_is_nan_when_both_raters_use_one_label_only():
    assert math.isnan(ag.cohen_kappa(list("NNNN"), list("NNNN")))


def test_kappa_three_classes_and_input_checks():
    assert ag.cohen_kappa(list("GTNG"), list("GTNG")) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ag.cohen_kappa(["Y"], ["Y", "N"])


def test_bootstrap_interval_is_reproducible_and_contains_the_estimate():
    a = list("YYYYNNNNNNYYNNNNNNNN" * 5)
    b = list("YYYNNNNNNNYYNNNNNYNN" * 5)
    ci = ag.bootstrap_kappa(a, b, n_boot=2000)
    assert ci == ag.bootstrap_kappa(a, b, n_boot=2000)
    assert ci[0] <= ag.cohen_kappa(a, b) <= ci[1]


def test_wilson_known_values():
    assert ag.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert ag.wilson(0, 10)[0] == 0.0 and ag.wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-3)


KEY = {"F1": {"condition_id": "c1", "question_id": "q1", "model": "qwen", "prompt": "plain", "needs_verdict": True},
       "F2": {"condition_id": "c2", "question_id": "q1", "model": "llama", "prompt": "plain", "needs_verdict": False},
       "F3": {"condition_id": "c3", "question_id": "q1", "model": "qwen", "prompt": "consistency", "needs_verdict": True},
       "F4": {"condition_id": "c4", "question_id": "q1", "model": "llama", "prompt": "consistency", "needs_verdict": False},
       "H1": {"condition_id": "c5", "question_id": "q1", "model": "qwen"}}


def test_pairs_skip_unclear_and_unparseable_and_respect_the_verdict_subset():
    labels = {"F1": {"flag": "Y", "verdict": "G"}, "F2": {"flag": "?"}, "F3": {"flag": "N", "verdict": "T"},
              "F4": {"flag": "N"}}
    judged = {("c1", "q1"): "Y", ("c2", "q1"): "N", ("c3", "q1"): None, ("c4", "q1"): "N"}
    rows, skipped = ag.pairs(labels, KEY, judged, "flag")
    assert [r[0] for r in rows] == ["F1", "F4"]
    assert skipped == {"human ?": 1, "judge missing/unparseable": 1}
    rows, _ = ag.pairs(labels, KEY, {("c1", "q1"): "G", ("c3", "q1"): "N"}, "verdict", only_verdict=True)
    assert [(r[0], r[1], r[2]) for r in rows] == [("F1", "G", "G"), ("F3", "T", "N")]


def test_internal_check_excludes_the_validation_sample():
    scored = [{"mode": "greedy", "outcome": "Correct", "condition_id": f"c{i}", "question_id": "q"} for i in range(4)]
    scored += [{"mode": "greedy", "outcome": "Poisoned", "condition_id": f"p{i}", "question_id": "q"} for i in range(2)]
    judged = {("c0", "q"): "G", ("c1", "q"): "G", ("c2", "q"): "N", ("c3", "q"): "G", ("p0", "q"): "T", ("p1", "q"): "G"}
    out = ag.internal_check(scored, judged, exclude={("c2", "q")})
    assert out["Correct"] == (3, 1.0) and out["Poisoned"] == (2, 0.5)


def test_hall_results_precision_and_miss_rate_by_model():
    key = {"H1": {"model": "qwen"}, "H2": {"model": "llama"}, "H3": {"model": "llama"},
           "M1": {"model": "mistral"}, "M2": {"model": "mistral"}}
    labels = {"H1": {"hall": "I"}, "H2": {"hall": "N"}, "H3": {"hall": "?"}, "M1": {"miss": "K"}, "M2": {"miss": "M"}}
    r = ag.hall_results(labels, key)
    assert (r["H"]["yes"], r["H"]["n"], r["H"]["unclear"]) == (1, 2, 1)
    assert r["H"]["by_model"]["llama"] == (0, 1) and r["M"]["by_model"]["mistral"] == (1, 2)