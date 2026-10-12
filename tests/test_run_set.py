"""Tests for the run script (D148-D149) with a fake GPU environment, tokenizer and model."""

import json

import pytest

from inragwetrust.conditions import MODELS, condition_table
from inragwetrust.models import run_set

A100_ENV = {"gpu": "NVIDIA A100-SXM4-40GB", "commit": "0" * 40, "commit_dirty": False}


class FakeGenerator:
    def __init__(self, model):
        self.model = model


@pytest.fixture
def calls(monkeypatch):
    """Replace GPU-dependent pieces with fakes; return the list of run_model calls."""
    recorded = []

    def fake_run_model(generator, tokenizer, rows, conditions, mode, out_dir, env, log=print):
        recorded.append({"model": generator.model, "rows": rows, "conditions": conditions,
                         "mode": mode, "out_dir": out_dir, "env": env})
        return [{"condition_id": c.condition_id, "status": "done", "prompts": len(rows),
                 "seconds": 0.1, "prompt_tokens": 10, "output_tokens": 5} for c in conditions]

    monkeypatch.setattr(run_set, "environment", lambda: dict(A100_ENV))
    monkeypatch.setattr(run_set, "load_tokenizer", lambda model: object())
    monkeypatch.setattr(run_set, "Generator", FakeGenerator)
    monkeypatch.setattr(run_set.runs, "run_model", fake_run_model)
    return recorded


def pilot(tmp_path, *extra):
    run_set.main(["--model", "qwen", "--split", "pilot", "--out", str(tmp_path), *extra])


def test_every_model_has_64_conditions():
    for model in MODELS:
        assert sum(c.model == model for c in condition_table()) == run_set.CONDITIONS_PER_MODEL


def test_pilot_runs_greedy_then_samples_on_50_questions_and_64_conditions(calls, tmp_path):
    pilot(tmp_path)
    assert [c["mode"] for c in calls] == ["greedy", "samples"]
    for call in calls:
        assert call["model"] == "qwen"
        assert len(call["rows"]) == 50 and {r["split"] for r in call["rows"]} == {"pilot"}
        assert len(call["conditions"]) == 64 and {c.model for c in call["conditions"]} == {"qwen"}
        assert call["out_dir"] == tmp_path and call["env"]["gpu"].startswith("NVIDIA A100")


def test_retrieval_and_closed_book_conditions_are_included(calls, tmp_path):
    pilot(tmp_path)
    runs_kinds = [c.run for c in calls[0]["conditions"]]
    assert runs_kinds.count("retrieval") == 12
    assert runs_kinds.count("closed_book") == 1


def test_one_run_log_per_mode(calls, tmp_path):
    pilot(tmp_path)
    logs = [json.loads(p.read_text()) for p in sorted((tmp_path / "_logs").glob("*.json"))]
    assert sorted(log["mode"] for log in logs) == ["greedy", "samples"]
    for log in logs:
        assert log["questions"] == 50 and len(log["shards"]) == 64
        assert "load_seconds" in log and "wall_seconds" in log


def test_modes_option_runs_only_the_chosen_mode(calls, tmp_path):
    pilot(tmp_path, "--modes", "greedy")
    assert [c["mode"] for c in calls] == ["greedy"]


def test_non_a100_gpu_is_refused_before_anything_runs(calls, monkeypatch, tmp_path):
    monkeypatch.setattr(run_set, "environment", lambda: dict(A100_ENV, gpu="NVIDIA L4"))
    with pytest.raises(SystemExit, match="D139"):
        pilot(tmp_path)
    assert calls == []


def test_unknown_model_is_rejected(calls, tmp_path):
    with pytest.raises(SystemExit):
        run_set.main(["--model", "qwen2", "--split", "pilot", "--out", str(tmp_path)])
    assert calls == []