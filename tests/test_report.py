"""Tests for the report tables (D4)."""

import csv

from inragwetrust.evaluation import report as rp
from inragwetrust.models.runs import write_atomic


def srow(cid, qid, outcome, run="poisoned", rho=0.5, ptype="swap", mode="greedy", sample=0, answer="Paris",
         hall=None, position="line-start", context=None):
    return {"condition_id": cid, "question_id": qid, "run": run, "model": "qwen", "prompt": "plain",
            "poison_type": ptype, "rho": rho, "context_id": context or f"{ptype}-{rho}", "mode": mode,
            "sample": sample, "n_tokens": 10, "finish_reason": "stop", "answer": answer, "position": position,
            "truncated": False, "cut": False, "outcome": outcome, "hall": hall}


SCORED = ([srow("c0", "q1", "Correct", run="clean", rho=0.0, ptype=None),
           srow("c0", "q2", "Correct", run="clean", rho=0.0, ptype=None),
           srow("c5", "q1", "Poisoned", answer="Lyon"),
           srow("c5", "q2", "OtherWrong", answer="Marseille", hall=True),
           srow("cb", "q1", "Correct", run="retrieval", rho=None, context="bm25-swap"),
           srow("cc", "q1", "OtherWrong", run="closed_book", rho=None, ptype=None, context="closedbook",
                answer="Rome", hall=True, position="mid-line")]
          + [srow("c5", "q1", o, mode="samples", sample=i, answer=a)
             for i, (o, a) in enumerate([("Poisoned", "Lyon")] * 4 + [("Correct", "Paris")])])
JUDGE = ([{"condition_id": c, "question_id": q, "task": "flag", "label": l}
          for c, q, l in (("c0", "q1", "N"), ("c0", "q2", "N"), ("c5", "q1", "Y"), ("c5", "q2", "Y"), ("cb", "q1", "N"))]
         + [{"condition_id": "c5", "question_id": "q2", "task": "verdict", "label": "G"}])
GOLDS = {"q1": "Paris", "q2": "Paris"}


def by(table, **match):
    return next(r for r in table if all(r[k] == v for k, v in match.items()))


def test_tables_and_values():
    t = rp.build_tables(SCORED, JUDGE, GOLDS)
    assert set(t) == {"by_rho", "by_rho_pooled", "grid", "retrieval", "closed_book", "sc", "format"}
    clean, half = by(t["by_rho"], level="0.00"), by(t["by_rho"], level="0.50")
    assert (clean["acc"], clean["acc_n"], clean["flag"]) == (1.0, 2, 0.0)
    assert (half["acc"], half["asr"], half["hall"], half["flag"]) == (0.0, 0.5, 0.5, 1.0)
    assert half["acc_plus"] == 0.5                                     # verdict G re-labels the OtherWrong
    assert by(t["retrieval"], context_id="bm25-swap")["acc"] == 1.0
    closed = by(t["closed_book"], model="qwen")
    assert (closed["flag"], closed["flag_n"], closed["hall"]) == ("", 0, 1.0)   # no flag without passages
    assert by(t["sc"], level="0.50")["sc"] == 0.8
    fmt = by(t["format"], mode="greedy")
    assert (fmt["n"], fmt["mid_line"]) == (6, round(1 / 6, 4))


def test_csv_files_have_interval_columns(tmp_path):
    paths = rp.write_tables(rp.build_tables(SCORED, JUDGE, GOLDS), tmp_path)
    assert len(paths) == 7
    header = next(csv.reader(open(tmp_path / "by_rho.csv", encoding="utf-8")))
    assert {"model", "prompt", "level", "acc", "acc_lo", "acc_hi", "acc_n", "acc_plus"} <= set(header)


def test_main_reads_the_run_folder_and_writes_tables(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    write_atomic(run / "_scored" / "scored.jsonl.gz", SCORED)
    write_atomic(run / "_judge" / "judge.jsonl.gz", JUDGE)
    monkeypatch.setattr(rp, "load_contexts", lambda split: [{"id": q, "gold_answer": g} for q, g in GOLDS.items()])
    rp.main(["--run", str(run), "--split", "pilot", "--out", str(tmp_path / "results")])
    assert (tmp_path / "results" / "grid.csv").exists()
    assert "Acc+ (exploratory)" in capsys.readouterr().out