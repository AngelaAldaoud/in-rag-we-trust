"""Tests for scoring a run folder (D2) on a tiny fake run."""

from inragwetrust.evaluation import score as scoring
from inragwetrust.models.runs import read_shard, write_atomic

ROWS = [{"id": "q1", "split": "pilot", "gold_answer": "Paris", "target_answer": "Lyon",
         "contexts": {"swap-0.50": [{"text": "Lyon is the capital of France."},
                                    {"text": "Paris is the capital of France."}],
                      "closedbook": []}}]


def record(mode, context_id, rho, texts, finish="stop"):
    return {"question_id": "q1", "split": "pilot", "condition_id": f"qwen__plain__{context_id}",
            "run": "poisoned" if rho else "closed_book", "model": "qwen", "prompt": "plain",
            "poison_type": "swap" if rho else None, "rho": rho, "context_id": context_id, "mode": mode,
            "completions": [{"text": t, "n_tokens": 7, "finish_reason": finish} for t in texts]}


def fake_run(tmp_path):
    write_atomic(tmp_path / "greedy/qwen/qwen__plain__swap-0.50.jsonl.gz",
                 [record("greedy", "swap-0.50", 0.5, ["Reasoning.\nAnswer: Paris"])])
    write_atomic(tmp_path / "samples/qwen/qwen__plain__swap-0.50.jsonl.gz",
                 [record("samples", "swap-0.50", 0.5,
                         ["Answer: Paris", "Answer: Lyon", "Answer: unknown", "Answer: Marseille", "no line"])])
    write_atomic(tmp_path / "greedy/qwen/qwen__plain__closedbook.jsonl.gz",
                 [record("greedy", "closedbook", None, ["Answer: Marseille"])])
    (tmp_path / "_logs").mkdir()
    (tmp_path / "_logs/qwen_greedy.json").write_text("{}")
    return tmp_path


def test_shard_paths_skip_logs_and_scored_output(tmp_path):
    run = fake_run(tmp_path)
    write_atomic(run / "_scored/scored.jsonl.gz", [{"x": 1}])
    assert [p.parent.parent.name for p in scoring.shard_paths(run)] == ["greedy", "greedy", "samples"]


def test_every_completion_gets_one_scored_row(tmp_path):
    rows = scoring.score_shards(scoring.shard_paths(fake_run(tmp_path)), ROWS)
    samples = [r for r in rows if r["mode"] == "samples"]
    assert len(rows) == 7 and [r["sample"] for r in samples] == [0, 1, 2, 3, 4]
    assert [r["outcome"] for r in samples] == ["Correct", "Poisoned", "Abstain", "OtherWrong", "NoAnswer"]
    assert samples[3]["hall"] is True and samples[0]["hall"] is None


def test_condition_fields_and_parse_results_are_carried(tmp_path):
    rows = scoring.score_shards(scoring.shard_paths(fake_run(tmp_path)), ROWS)
    greedy = next(r for r in rows if r["context_id"] == "swap-0.50" and r["mode"] == "greedy")
    assert {k: greedy[k] for k in ("question_id", "model", "prompt", "poison_type", "rho", "answer",
                                    "position", "truncated", "outcome")} == {
        "question_id": "q1", "model": "qwen", "prompt": "plain", "poison_type": "swap", "rho": 0.5,
        "answer": "Paris", "position": "line-start", "truncated": False, "outcome": "Correct"}


def test_closed_book_uses_no_passages(tmp_path):
    rows = scoring.score_shards(scoring.shard_paths(fake_run(tmp_path)), ROWS)
    closed = next(r for r in rows if r["context_id"] == "closedbook")
    assert (closed["outcome"], closed["hall"], closed["hall_strict"]) == ("OtherWrong", True, True)


def test_main_writes_the_scored_file_and_a_summary(tmp_path, monkeypatch, capsys):
    run = fake_run(tmp_path)
    monkeypatch.setattr(scoring, "load_contexts", lambda split: ROWS)
    scoring.main(["--run", str(run), "--split", "pilot"])
    written = read_shard(run / "_scored/scored.jsonl.gz")
    assert len(written) == 7
    assert "3 shards, 7 completions scored" in capsys.readouterr().out