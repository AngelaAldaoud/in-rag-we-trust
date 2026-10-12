"""Gemma judge: conflict flag (D48) and answer verdict (D160, D164). Runs in its own process (D148).

    python -u -m inragwetrust.evaluation.judge check --run <run dir> --split pilot --gemma-dir <dir> --exclude <key.json>
    python -u -m inragwetrust.evaluation.judge run   --run <run dir> --split pilot --gemma-dir <dir>
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # D102, set before vLLM is imported

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from inragwetrust.data.build import SEED
from inragwetrust.evaluation.parse import parse_answer
from inragwetrust.models.runs import read_shard
from inragwetrust.retrieval import build as files
from inragwetrust.retrieval.load import SPLITS, load_contexts

JUDGE_MODEL = "kaggle:google/gemma-2/transformers/gemma-2-9b-it/2"     # same weights as Block B
JUDGE_MAX_LEN, JUDGE_MAX_TOKENS, JUDGE_MEMORY = 2048, 4, 0.95           # B4 settings (D119)
JUDGE_DIR, JUDGE_FILE, JUDGE_META = "_judge", "judge.jsonl.gz", "judge_meta.json"
REPO = Path(__file__).resolve().parents[3]

FLAG_PROMPT = """You are checking a response written by an AI assistant that was given several passages and a question.

Question: Does this response explicitly say that the passages disagree, or that some of them may be unreliable?

Answer "yes" if the response says, in any words, that the passages disagree, contradict each other or give different answers, or that some passage or source may be unreliable, wrong or doubtful.
Answer "no" if it does not. Also answer "no" if it only says that information is missing, if it simply relies on some passages without saying that others differ, or if it says that the passages agree.

Response:
<<<
{response}
>>>

Reply with exactly one word: yes or no."""

VERDICT_PROMPT = """You are grading a short answer to a question.

Question: {question}
Gold answer: {gold}
Other answer: {target}
Answer to grade: {answer}

Does the answer to grade give the gold answer, the other answer, or neither?
Count it as the same answer if it refers to the same thing in a different form: an abbreviation, a translation, a shorter or longer name, or a differently written date or number. If it refers to a different thing, it is neither.

Reply with exactly one word: gold, other, or neither."""

_WORD = re.compile(r"[a-z]+")


def first_word(text: str) -> str:
    m = _WORD.search(text.lower())
    return m.group(0) if m else ""


def parse_flag(text: str) -> str | None:
    return {"yes": "Y", "no": "N"}.get(first_word(text))


def parse_verdict(text: str) -> str | None:
    return {"gold": "G", "other": "T", "neither": "N"}.get(first_word(text))


def prompt_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(p.encode()).hexdigest()[:12]
            for name, p in (("flag", FLAG_PROMPT), ("verdict", VERDICT_PROMPT))}


def greedy_records(run_dir):
    for path in sorted(Path(run_dir, "greedy").glob("*/*.jsonl.gz")):
        yield from read_shard(path)


def build_tasks(records, rows) -> list[dict]:
    """A flag task per greedy response with passages; a verdict task per greedy response with an answer."""
    by_id = {r["id"]: r for r in rows}
    tasks = []
    for rec in records:
        completion = rec["completions"][0]
        base = {k: rec[k] for k in ("condition_id", "question_id", "model", "prompt", "run")}
        if rec["run"] != "closed_book":
            tasks.append(base | {"task": "flag",
                                 "text": FLAG_PROMPT.format(response=completion["text"].strip())})
        answer = parse_answer(completion["text"], completion["finish_reason"]).answer
        if answer is not None:
            q = by_id[rec["question_id"]]
            tasks.append(base | {"task": "verdict", "text": VERDICT_PROMPT.format(
                question=q["question"], gold=q["gold_answer"], target=q["target_answer"], answer=answer)})
    return tasks


def render(tokenizer, text: str) -> list[int]:
    """Gemma chat template, tokenised once without extra special tokens (as D129)."""
    chat = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(chat, add_special_tokens=False)
    if len(ids) > JUDGE_MAX_LEN - JUDGE_MAX_TOKENS:
        raise ValueError(f"judge prompt has {len(ids)} tokens, limit {JUDGE_MAX_LEN - JUDGE_MAX_TOKENS}")
    return ids


def judge(engine, params_factory, tokenizer, tasks) -> list[dict]:
    """Greedy one-word verdicts; the raw output is kept next to the parsed label (None if unparseable)."""
    prompts = [{"prompt_token_ids": render(tokenizer, t["text"])} for t in tasks]
    params = params_factory(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, seed=SEED)
    outputs = engine.generate(prompts, params, use_tqdm=False)
    if len(outputs) != len(tasks):
        raise RuntimeError("the engine returned a different number of outputs")
    results = []
    for t, out in zip(tasks, outputs):
        raw = out.outputs[0].text
        label = parse_flag(raw) if t["task"] == "flag" else parse_verdict(raw)
        results.append({k: t[k] for k in ("condition_id", "question_id", "model", "prompt", "run", "task")}
                       | {"raw": raw, "label": label})
    return results


def load_judge(gemma_dir: str):
    """Gemma exactly as in Block B4: bf16, 2048 tokens, 95% memory, project seed, D102 + D119."""
    from vllm import LLM, SamplingParams

    notebook_out, notebook_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        engine = LLM(model=gemma_dir, dtype="bfloat16", max_model_len=JUDGE_MAX_LEN,
                     gpu_memory_utilization=JUDGE_MEMORY, seed=SEED)
    finally:
        sys.stdout, sys.stderr = notebook_out, notebook_err
    return engine, SamplingParams, engine.get_tokenizer()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--gemma-dir", required=True)
    parser.add_argument("--exclude", type=Path, help="validation key whose items 'check' must not use")
    args = parser.parse_args(argv)

    tasks = build_tasks(list(greedy_records(args.run)), load_contexts(args.split))
    if args.action == "check":
        excluded = set()
        if args.exclude:
            excluded = {(v["condition_id"], v["question_id"]) for v in json.loads(args.exclude.read_text()).values()}
        rng = random.Random(SEED)
        pool = [t for t in tasks if (t["condition_id"], t["question_id"]) not in excluded]
        tasks = (rng.sample([t for t in pool if t["task"] == "flag"], 10)
                 + rng.sample([t for t in pool if t["task"] == "verdict"], 10))
    print(f"{len(tasks)} judge tasks | prompts {prompt_hashes()}", flush=True)

    start = time.time()
    engine, params_factory, tokenizer = load_judge(args.gemma_dir)
    load_seconds = round(time.time() - start, 1)
    start = time.time()
    results = judge(engine, params_factory, tokenizer, tasks)
    seconds = round(time.time() - start, 1)
    print(f"loaded in {load_seconds} s, judged in {seconds} s", flush=True)

    if args.action == "check":
        print("\n--- first flag prompt ---\n" + tasks[0]["text"][:1500])
        for t, r in zip(tasks, results):
            print(f"{r['task']:8s} {r['model']:8s} raw={r['raw']!r:14s} -> {r['label']}")
        return

    out = args.run / JUDGE_DIR
    out.mkdir(parents=True, exist_ok=True)
    files.write_jsonl_gz(out / JUDGE_FILE, results)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (out / JUDGE_META).write_text(json.dumps({
        "judge_model": JUDGE_MODEL, "gemma_dir": args.gemma_dir, "max_model_len": JUDGE_MAX_LEN,
        "max_tokens": JUDGE_MAX_TOKENS, "memory": JUDGE_MEMORY, "seed": SEED, "prompts": prompt_hashes(),
        "commit": commit, "tasks": len(results), "load_seconds": load_seconds, "seconds": seconds}, indent=2))
    for task in ("flag", "verdict"):
        labels = [r["label"] for r in results if r["task"] == task]
        counts = {k: labels.count(k) for k in sorted({str(x) for x in labels})}
        print(f"{task:8s} n={len(labels)} labels={ {str(k): labels.count(k if k != 'None' else None) for k in counts} }")
    print(f"wrote {out / JUDGE_FILE}")


if __name__ == "__main__":
    main()