"""Gemma judge: conflict flag (D48) and answer verdict (D160, D164, D165). Runs in its own process (D148).

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

# D165: symmetric references A/B (seeded order), no 'gold'/'other' wording, sameness only
VERDICT_PROMPT = """You are comparing a short answer with two reference answers, A and B.

Question: {question}
Reference A: {ref_a}
Reference B: {ref_b}
Answer to compare: {answer}

Which reference does the answer to compare refer to? Judge only whether it names the same thing as a reference, possibly in a different form: an abbreviation, a translation, a shorter or longer name, or a differently written date or number. Do not judge whether any answer is true, and do not use your own knowledge of the correct answer. If the answer refers to something different from both references, it is neither.

Reply with exactly one word: A, B, or neither."""

_WORD = re.compile(r"[a-z]+")


def first_word(text: str) -> str:
    m = _WORD.search(text.lower())
    return m.group(0) if m else ""


def parse_flag(text: str) -> str | None:
    return {"yes": "Y", "no": "N"}.get(first_word(text))


def parse_verdict(text: str, gold_is_a: bool) -> str | None:
    """A/B mapped back to gold (G) or target (T) through the item's recorded order; neither -> N."""
    word = first_word(text)
    if word == "neither":
        return "N"
    if word in ("a", "b"):
        return "G" if (word == "a") == gold_is_a else "T"
    return None


def gold_is_a(condition_id: str, question_id: str) -> bool:
    """Seeded coin flip per item: is the gold answer shown as reference A? (D165)"""
    return int(hashlib.sha256(f"{SEED}-{condition_id}-{question_id}-order".encode()).hexdigest(), 16) % 2 == 0


def prompt_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(p.encode()).hexdigest()[:12]
            for name, p in (("flag", FLAG_PROMPT), ("verdict", VERDICT_PROMPT))}


def greedy_records(run_dir):
    for path in sorted(Path(run_dir, "greedy").glob("*/*.jsonl.gz")):
        yield from read_shard(path)


def build_tasks(records, rows, tasks_wanted=("flag", "verdict")) -> list[dict]:
    """A flag task per greedy response with passages; a verdict task per greedy response with an answer."""
    by_id = {r["id"]: r for r in rows}
    tasks = []
    for rec in records:
        completion = rec["completions"][0]
        base = {k: rec[k] for k in ("condition_id", "question_id", "model", "prompt", "run")}
        if "flag" in tasks_wanted and rec["run"] != "closed_book":
            tasks.append(base | {"task": "flag",
                                 "text": FLAG_PROMPT.format(response=completion["text"].strip())})
        answer = parse_answer(completion["text"], completion["finish_reason"]).answer
        if "verdict" in tasks_wanted and answer is not None:
            q = by_id[rec["question_id"]]
            a_first = gold_is_a(rec["condition_id"], rec["question_id"])
            ref_a, ref_b = ((q["gold_answer"], q["target_answer"]) if a_first
                            else (q["target_answer"], q["gold_answer"]))
            tasks.append(base | {"task": "verdict", "gold_is_a": a_first, "text": VERDICT_PROMPT.format(
                question=q["question"], ref_a=ref_a, ref_b=ref_b, answer=answer)})
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
        if t["task"] == "flag":
            extra, label = {}, parse_flag(raw)
        else:
            extra, label = {"gold_is_a": t["gold_is_a"]}, parse_verdict(raw, t["gold_is_a"])
        results.append({k: t[k] for k in ("condition_id", "question_id", "model", "prompt", "run", "task")}
                       | extra | {"raw": raw, "label": label})
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
        print("\n--- first verdict prompt ---\n" + next(t["text"] for t in tasks if t["task"] == "verdict"))
        for r in results:
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
        labels = [str(r["label"]) for r in results if r["task"] == task]
        print(f"{task:8s} n={len(labels)} labels={ {k: labels.count(k) for k in sorted(set(labels))} }")
    print(f"wrote {out / JUDGE_FILE}")


if __name__ == "__main__":
    main()