"""Tests for the D3 validation sample and sheet (D163)."""

import pytest

from inragwetrust.evaluation import validation as v

MODELS, PROMPTS = ("qwen", "mistral", "llama"), ("plain", "consistency", "reliability")


def row(model, prompt, lvl, i, outcome="Correct", hall=None, mode="greedy"):
    retrieval = lvl == "bm25"
    return {"model": model, "prompt": prompt, "mode": mode, "question_id": f"q{i % 4}",
            "condition_id": f"{model}__{prompt}__{lvl}__{i}", "run": "retrieval" if retrieval else "poisoned",
            "rho": None if retrieval else lvl, "context_id": "swap-0.50", "outcome": outcome,
            "answer": "Marseille", "hall": hall, "missing_facts": ["Marseille"] if hall else None}


SCORED = ([row(m, p, lvl, i, "OtherWrong" if i % 3 == 0 else "Correct", True if i % 3 == 0 else None)
           for m in MODELS for p in PROMPTS for lvl in v.LEVELS for i in range(12)]
          + [dict(row("qwen", "plain", 0.0, i), run="closed_book") for i in range(20)]
          + [row("qwen", "plain", 0.0, i, mode="samples") for i in range(20)])
ROWS = [{"id": f"q{i}", "question": f"Question {i}?", "gold_answer": "Paris", "target_answer": "Lyon",
         "contexts": {"swap-0.50": [{"text": "Lyon is in France."}, {"text": "Paris is in France."}]}}
        for i in range(4)]


def test_draw_takes_five_per_cell_then_fills_to_300():
    chosen, hall = v.draw(SCORED)
    assert len(chosen) == 300 and len({v.key_of(r) for r in chosen}) == 300
    cells = {}
    for r in chosen:
        cells[(r["model"], r["prompt"], str(v.level(r)))] = cells.get((r["model"], r["prompt"], str(v.level(r))), 0) + 1
    assert len(cells) == 54 and min(cells.values()) >= 5
    assert all(r["mode"] == "greedy" and r["run"] != "closed_book" for r in chosen)


def test_hall_items_are_flagged_other_wrong_ten_per_model_and_disjoint():
    chosen, hall = v.draw(SCORED)
    assert sorted(r["model"] for r in hall) == ["llama"] * 10 + ["mistral"] * 10 + ["qwen"] * 10
    assert all(r["outcome"] == "OtherWrong" and r["hall"] for r in hall)
    assert not {v.key_of(r) for r in hall} & {v.key_of(r) for r in chosen}


def test_draw_is_reproducible():
    assert v.draw(SCORED) == v.draw(SCORED)


def test_sheet_round_trip(tmp_path):
    chosen, hall = v.draw(SCORED)
    texts = {v.key_of(r): "Passage [1] says Lyon.\nAnswer: Marseille" for r in chosen}
    v.write_sheet(tmp_path, chosen, hall, texts, ROWS)
    sheet = (tmp_path / v.SHEET).read_text()
    assert "| Passage [1] says Lyon." in sheet and "facts found in no passage: Marseille" in sheet
    filled = sheet.replace("flag: \n", "flag: y\n").replace("verdict: \n", "verdict: n\n").replace("hall: \n", "hall: i\n")
    (tmp_path / v.SHEET).write_text(filled)
    labels = v.read_fields(tmp_path / v.SHEET)
    n_verdict = sum(r["outcome"] == "OtherWrong" for r in chosen)
    assert sum("flag" in x for x in labels.values()) == 300
    assert sum("verdict" in x for x in labels.values()) == n_verdict
    assert sum(x.get("hall") == "I" for x in labels.values()) == 30


def test_sheet_command_refuses_to_overwrite(tmp_path):
    (tmp_path / v.SHEET).write_text("my labels")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        v.main(["sheet", "--run", "runs/pilot", "--split", "pilot", "--out", str(tmp_path)])