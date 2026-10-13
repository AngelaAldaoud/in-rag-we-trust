import pandas as pd
import pytest

from inragwetrust.analysis import audit


def row(qid, ctx, outcome, verdict=None, question="Where was he born?", gold="Paris", target="Lyon"):
    return {"question_id": qid, "context_id": ctx, "outcome": outcome, "verdict_label": verdict,
            "model": "m", "prompt": "plain", "answer": "x", "question": question,
            "gold_answer": gold, "target_answer": target}


def table():
    return pd.DataFrame([
        row("q1", "closedbook", "Poisoned"), row("q1", "clean-0.00", "Correct"),
        row("q2", "clean-0.00", "Correct", question="Who is older, Ann Lee or Bob Ray?", gold="Ann Lee"),
        row("q3", "clean-0.00", "Correct"), row("q3", "swap-1.00", "Poisoned"),
        row("q4", "clean-0.00", "OtherWrong", verdict="T"),
        row("q5", "clean-0.00", "Correct", verdict="T"),
    ])


def test_candidates_signals_and_reasons():
    c = audit.candidates(table()).set_index("question_id")
    assert sorted(c.index) == ["q1", "q2", "q4"]
    assert c.loc["q1", "reason"] == "gold_error" and c.loc["q1", "cb_poisoned"] == 1
    assert c.loc["q2", "reason"] == "comparison" and c.loc["q2", "gold_in_question"]
    assert c.loc["q4", "baseline_verdict_t"] == 1 and "clean-0.00" in c.loc["q4", "answers"]


def test_sheet_round_trip_and_protection(tmp_path):
    p = tmp_path / "sheet.txt"
    audit.write_sheet(audit.candidates(table()), p)
    with pytest.raises(FileExistsError):
        audit.write_sheet(audit.candidates(table()), p)
    with pytest.raises(ValueError, match="must be yes or no"):
        audit.read_sheet(p)
    text = p.read_text().replace("\ngold_error: \n", "\ngold_error: no\n").replace("\ncomparison: \n", "\ncomparison: yes\n")
    p.write_text(text)
    with pytest.raises(ValueError, match="note is required"):
        audit.read_sheet(p)
    p.write_text(text.replace("\nnote: \n", "\nnote: names two people\n"))
    dec = audit.read_sheet(p)
    assert len(dec) == 3 and dec["comparison"].all() and not dec["gold_error"].any()
    assert set(dec["reason"]) == {"gold_error", "comparison"}