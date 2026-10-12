"""Shards, schema and resume (C3, D136-D137), with fake tokenizer and generator."""

import pytest

from inragwetrust.conditions import condition_table
from inragwetrust.models import runs
from inragwetrust.models.generator import MODES, Completion


class FakeTokenizer:
    bos_token_id = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text[:10]]}


class FakeGenerator:
    model = "qwen"

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def generate(self, requests, mode):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated disconnect")
        return [[Completion(f"Answer: {r.question_id}-{k}", 3, "stop")
                 for k in range(MODES[mode]["n"])] for r in requests]


ROWS = [{"id": q, "split": "pilot", "question": f"question {q}?",
         "contexts": {"closedbook": [], "clean-0.00": [{"text": "passage A"}, {"text": "passage B"}]}}
        for q in ("q1", "q2")]
TABLE = {c.condition_id: c for c in condition_table()}
CLOSED, CLEAN = TABLE["qwen__plain__closedbook"], TABLE["qwen__plain__clean-0.00"]
ENV = {"vllm": "test"}


def test_schema_and_order(tmp_path):
    runs.run_model(FakeGenerator(), FakeTokenizer(), ROWS, [CLOSED, CLEAN], "greedy", tmp_path, ENV,
                   log=lambda _: None)
    recs = runs.read_shard(runs.shard_path(tmp_path, CLEAN, "greedy"))
    assert [r["question_id"] for r in recs] == ["q1", "q2"]
    r = recs[0]
    assert r["condition_id"] == "qwen__plain__clean-0.00" and r["rho"] == 0.0
    assert r["temperature"] == 0.0 and r["n"] == 1 and r["max_tokens"] == 512
    assert len(r["prompt_hash"]) == 64 and r["env"] == ENV
    assert r["completions"] == [{"text": "Answer: q1-0", "n_tokens": 3, "finish_reason": "stop"}]


def test_samples_have_five_completions(tmp_path):
    runs.run_shard(FakeGenerator(), FakeTokenizer(), ROWS, CLEAN, "samples", tmp_path, ENV)
    recs = runs.read_shard(runs.shard_path(tmp_path, CLEAN, "samples"))
    assert all(len(r["completions"]) == 5 and r["temperature"] == 0.7 for r in recs)


def test_resume_skips_finished_shards(tmp_path):
    gen = FakeGenerator()
    first = runs.run_model(gen, FakeTokenizer(), ROWS, [CLOSED, CLEAN], "greedy", tmp_path, ENV,
                           log=lambda _: None)
    second = runs.run_model(gen, FakeTokenizer(), ROWS, [CLOSED, CLEAN], "greedy", tmp_path, ENV,
                            log=lambda _: None)
    assert [s["status"] for s in first] == ["done", "done"]
    assert [s["status"] for s in second] == ["skipped", "skipped"] and gen.calls == 2


def test_interrupted_write_is_redone(tmp_path, monkeypatch):
    def crash_after_tmp(path, records):          # simulate a disconnect during writing
        path.write_text("partial")
        raise RuntimeError("disconnect")

    monkeypatch.setattr(runs.files, "write_jsonl_gz", crash_after_tmp)
    with pytest.raises(RuntimeError):
        runs.run_shard(FakeGenerator(), FakeTokenizer(), ROWS, CLEAN, "greedy", tmp_path, ENV)
    final = runs.shard_path(tmp_path, CLEAN, "greedy")
    assert not final.exists() and final.with_name(final.name + ".tmp").exists()

    monkeypatch.undo()
    s = runs.run_shard(FakeGenerator(), FakeTokenizer(), ROWS, CLEAN, "greedy", tmp_path, ENV)
    assert s["status"] == "done" and len(runs.read_shard(final)) == 2


def test_missing_bm25_context_is_a_clear_error(tmp_path):
    with pytest.raises(KeyError, match="C4"):
        runs.run_shard(FakeGenerator(), FakeTokenizer(), ROWS,
                       TABLE["qwen__plain__bm25-swap"], "greedy", tmp_path, ENV)