"""Tests for the D2 hand audit (D159)."""

import json

import pytest

from inragwetrust.evaluation import audit


def row(i, outcome, model="qwen", answer="Paris", hall=None, context="swap-0.50", mode="greedy"):
    return {"question_id": f"q{i % 3}", "condition_id": f"c{i}", "model": model, "mode": mode,
            "outcome": outcome, "answer": answer, "hall": hall, "context_id": context}


SCORED = ([row(i, "Correct") for i in range(30)] + [row(i, "Poisoned") for i in range(30)]
          + [row(i, "Both") for i in range(3)] + [row(i, "OtherWrong", m, "Marseille", True)
                                                  for i in range(30) for m in ("qwen", "mistral", "llama")]
          + [row(i, "Abstain", answer="unknown") for i in range(30)]
          + [row(i, "Abstain", answer="not mentioned in passages") for i in range(30)]
          + [row(i, "NoAnswer", answer=None) for i in range(30)]
          + [row(i, "Correct", mode="samples") for i in range(30)])

ROWS = [{"id": f"q{i}", "question": f"Question {i}?", "gold_answer": "Paris", "target_answer": "Lyon",
         "contexts": {"swap-0.50": [{"text": "Lyon and Paris are cities in France."}]}} for i in range(3)]


def test_draw_respects_quotas_and_excludes_samples_and_no_answer():
    items, hall = audit.draw(SCORED)
    strata = [audit.stratum(r) for r in items]
    assert {s: strata.count(s) for s in audit.OUTCOME_QUOTA} == {
        "Correct": 10, "Poisoned": 10, "Both": 3, "OtherWrong": 20, "Abstain-unknown": 5, "Abstain-phrase": 10}
    assert all(r["mode"] == "greedy" and r["outcome"] != "NoAnswer" for r in items)
    assert sorted(r["model"] for r in hall) == ["llama"] * 5 + ["mistral"] * 10 + ["qwen"] * 5


def test_draw_is_reproducible():
    assert audit.draw(SCORED) == audit.draw(SCORED)


def test_sheet_is_blind_and_round_trips_with_labels(tmp_path):
    items, hall = audit.draw(SCORED)
    audit.write_sheet(tmp_path, items, hall, ROWS)
    sheet = (tmp_path / audit.SHEET).read_text()
    assert "Correct" not in sheet.split("### H1")[0].split("# Set H")[1]       # no outcome names in set O
    assert "words in no passage: marseille" in sheet
    labelled = sheet.replace("label: ", "label: c")
    (tmp_path / audit.SHEET).write_text(labelled)
    labels = audit.read_labels(tmp_path / audit.SHEET)
    key = json.loads((tmp_path / audit.KEY).read_text())
    assert set(labels) == set(key) and set(labels.values()) == {"C"}


def test_results_refuse_unlabelled_or_invalid_items():
    key = {"O1": {"outcome": "Correct", "stratum": "Correct", "model": "qwen"},
           "H1": {"outcome": "OtherWrong", "model": "mistral", "hall": True}}
    with pytest.raises(ValueError, match="unlabelled"):
        audit.results({"O1": "C", "H1": ""}, key)
    with pytest.raises(ValueError, match="invalid"):
        audit.results({"O1": "X", "H1": "I"}, key)


def test_results_count_agreement_confusion_and_hall_precision():
    key = {"O1": {"outcome": "Correct", "stratum": "Correct", "model": "qwen"},
           "O2": {"outcome": "OtherWrong", "stratum": "OtherWrong", "model": "mistral"},
           "H1": {"outcome": "OtherWrong", "model": "mistral", "hall": True}}
    r = audit.results({"O1": "C", "O2": "C", "H1": "N"}, key)
    assert r["agree"][("Correct", True)] == 1 and r["agree"][("OtherWrong", False)] == 1
    assert r["confusion"][("OtherWrong", "Correct")] == 1
    assert r["hall"][("mistral", "not invented")] == 1


def test_sheet_command_refuses_to_overwrite(tmp_path):
    (tmp_path / audit.SHEET).write_text("labels I typed")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        audit.main(["sheet", "--run", "runs/pilot", "--split", "pilot", "--out", str(tmp_path)])