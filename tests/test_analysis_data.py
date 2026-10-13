import gzip
import json

import pandas as pd
import pytest

from inragwetrust.analysis import data
from inragwetrust.poisoning.poisons import POISON_TYPES


def write_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def resp(cid, qid="q1", mode="greedy", outcome="Correct"):
    return {"condition_id": cid, "question_id": qid, "mode": mode, "outcome": outcome,
            "context_id": cid.split("__")[-1], "model": "m"}


def judge_row(cid, task, label, qid="q1"):
    return {"condition_id": cid, "question_id": qid, "task": task, "label": label}


INFO = pd.DataFrame([{"question_id": "q1", "hotpot_type": "bridge"}])
LEAKS = pd.DataFrame([{"question_id": "q1", "leak_number": False, "leak_word": True}])
EFF = pd.DataFrame([{"question_id": "q1", "eff_rho_swap": 0.6}])


def test_load_greedy_keeps_greedy_and_rejects_duplicates(tmp_path):
    p = tmp_path / "s.jsonl.gz"
    write_gz(p, [resp("a"), resp("a", mode="samples"), resp("b")])
    assert len(data.load_greedy(p)) == 2
    write_gz(p, [resp("a"), resp("a")])
    with pytest.raises(ValueError, match="duplicate"):
        data.load_greedy(p)


def test_build_table_joins_labels_and_effective_rho():
    greedy = pd.DataFrame([resp("m__plain__swap-0.25"), resp("m__plain__closedbook"),
                           resp("m__plain__bm25-swap")])
    judge = pd.DataFrame([judge_row("m__plain__swap-0.25", "flag", "Y"),
                          judge_row("m__plain__bm25-swap", "flag", "N"),
                          judge_row("m__plain__swap-0.25", "verdict", None)])
    df = data.build_table(greedy, judge, INFO, LEAKS, EFF).set_index("context_id")
    assert df.loc["swap-0.25", "flag"] == 1.0 and df.loc["bm25-swap", "flag"] == 0.0
    assert not df.loc["closedbook", "flag_judged"] and pd.isna(df.loc["closedbook", "flag"])
    assert df.loc["swap-0.25", "verdict_judged"] and pd.isna(df.loc["swap-0.25", "verdict_label"])
    assert not df.loc["closedbook", "verdict_judged"]
    assert df.loc["bm25-swap", "eff_rho"] == 0.6 and pd.isna(df.loc["swap-0.25", "eff_rho"])
    assert bool(df.loc["swap-0.25", "leak_word"]) and df.loc["swap-0.25", "hotpot_type"] == "bridge"
    assert "eff_rho_swap" not in df.columns


def test_build_table_rejects_bad_judge_rows():
    greedy = pd.DataFrame([resp("m__plain__swap-0.25")])
    with pytest.raises(ValueError, match="unexpected flag"):
        data.build_table(greedy, pd.DataFrame([judge_row("m__plain__swap-0.25", "flag", "maybe")]),
                         INFO, LEAKS, EFF)
    with pytest.raises(ValueError, match="without a greedy response"):
        data.build_table(greedy, pd.DataFrame([judge_row("other", "flag", "Y")]), INFO, LEAKS, EFF)


def test_build_table_rejects_missing_question():
    greedy = pd.DataFrame([resp("m__plain__swap-0.25", qid="q2")])
    judge = pd.DataFrame(columns=["condition_id", "question_id", "task", "label"])
    with pytest.raises(ValueError, match="missing from the info table"):
        data.build_table(greedy, judge, INFO, LEAKS, EFF)


def test_leak_flags_uses_b4_rule(tmp_path):
    def rec(qid, text):
        return {"id": qid, "gold_answer": "Paris", "target_answer": "Lyon", "clean": ["It is in Paris."],
                "poisoned": {p.name: [text] for p in POISON_TYPES}}
    p = tmp_path / "p.jsonl.gz"
    write_gz(p, [rec("leaky", "It is in Lyon, near Paris."), rec("clean", "It is in Lyon."), rec("other", "x")])
    out = data.leak_flags(p, {"leaky", "clean"}).set_index("question_id")
    assert out.loc["leaky", "leak_word"] and not out.loc["clean", "leak_word"]
    assert len(out) == 2


def test_question_info_uses_contexts_and_official_type(tmp_path):
    cp, qp, rp = tmp_path / "c.jsonl.gz", tmp_path / "q.jsonl.gz", tmp_path / "raw.parquet"
    write_gz(cp, [{"id": "q1", "split": "main", "question": "?", "gold_answer": "a", "target_answer": "final"},
                  {"id": "q9", "split": "pilot", "question": "?", "gold_answer": "a", "target_answer": "x"}])
    write_gz(qp, [{"id": "q1", "split": "reserve", "answer_type": "person", "level": "hard",
                   "target_answer": "old"}])
    pd.DataFrame([{"id": "q1", "type": "comparison"}, {"id": "q9", "type": "bridge"}]).to_parquet(rp)
    info = data.question_info(cp, qp, rp, {"q1"})
    assert info["question_id"].tolist() == ["q1"]
    assert info["target_answer"].tolist() == ["final"] and info["hotpot_type"].tolist() == ["comparison"]
    with pytest.raises(ValueError, match="missing from the questions file"):
        data.question_info(cp, qp, rp, {"q1", "q9"})


def test_check_main_reports_wrong_counts():
    df = pd.DataFrame({"question_id": ["q1"], "flag_judged": [True], "flag": [1.0],
                       "verdict_judged": [False], "verdict_label": [None]})
    exp = {"responses": 1, "questions": 1, "flags": 1, "flags_unparseable": 0,
           "verdicts": 0, "verdicts_unparseable": 0}
    assert data.check_main(df, exp)["flags"] == 1
    with pytest.raises(ValueError, match="count check failed"):
        data.check_main(df, {**exp, "responses": 2})