"""Assembly of the greedy main-run responses for Block F (analysis plan, sections 1-2).

One row per greedy response, with its rule outcome, judge flag and verdict, question
information, the official HotpotQA question type, B4 leak flags and, for BM25 rows,
the per-question effective rho (D144). The main set is defined by the responses
themselves; question text, gold and target come from the contexts file the models saw.
Build with:
    uv run python -m inragwetrust.analysis.data
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.poisoning.build import read_jsonl_gz
from inragwetrust.poisoning.checks import check_poison
from inragwetrust.poisoning.poisons import POISON_TYPES

RUN_DIR = Path("runs/main")
RAW_HOTPOT = Path("data/raw/hotpot_dev_distractor.parquet")
OUT = RUN_DIR / "_analysis" / "greedy.parquet"
KEYS = ["condition_id", "question_id"]
FLAG_VALUES = {"Y": 1.0, "N": 0.0}
VERDICT_LABELS = {"G", "T", "N"}
MAIN_COUNTS = {"responses": 96_000, "questions": 500, "flags": 94_500, "flags_unparseable": 0,
               "verdicts": 93_547, "verdicts_unparseable": 32}


def load_greedy(path) -> pd.DataFrame:
    """Greedy rows of the scored file; one row per (condition, question)."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["mode"] == "greedy":
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.duplicated(KEYS).any():
        raise ValueError("duplicate greedy responses")
    return df


def task_labels(judge: pd.DataFrame, task: str, allowed: set[str]) -> pd.DataFrame:
    """Judge labels of one task, checked for duplicates and unexpected values."""
    part = judge.loc[judge["task"] == task, KEYS + ["label"]]
    if part.duplicated(KEYS).any():
        raise ValueError(f"duplicate {task} judgements")
    unexpected = set(part["label"].dropna()) - allowed
    if unexpected:
        raise ValueError(f"unexpected {task} labels: {sorted(unexpected)}")
    return part.rename(columns={"label": f"{task}_label"}).assign(**{f"{task}_judged": True})


def _unique(rows: list[dict], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or df.duplicated("question_id").any():
        raise ValueError(f"{name}: no rows or duplicate questions")
    return df


def question_info(contexts_path, questions_path, raw_path, ids: set[str]) -> pd.DataFrame:
    """Question, gold and target as the models saw them; answer type, level, official HotpotQA type."""
    ctx = _unique([{"question_id": r["id"], "question": r["question"], "gold_answer": r["gold_answer"],
                    "target_answer": r["target_answer"]}
                   for r in read_jsonl_gz(contexts_path) if r["id"] in ids], "contexts")
    qs = _unique([{"question_id": r["id"], "answer_type": r["answer_type"], "level": r["level"]}
                  for r in read_jsonl_gz(questions_path) if r["id"] in ids], "questions")
    raw = pd.read_parquet(raw_path, columns=["id", "type"])
    raw = raw[raw["id"].isin(ids)].rename(columns={"id": "question_id", "type": "hotpot_type"})
    info = ctx
    for name, part in (("questions", qs), ("raw HotpotQA", raw)):
        missing = set(info["question_id"]) - set(part["question_id"])
        if missing:
            raise ValueError(f"{len(missing)} questions missing from the {name} file")
        info = info.merge(part, on="question_id", how="left", validate="one_to_one")
    return info


def leak_flags(poisons_path, ids: set[str]) -> pd.DataFrame:
    """Per question: does any poisoned passage keep a number or a key word of the gold (B4 rule)?"""
    out, seen = [], set()
    for rec in read_jsonl_gz(poisons_path):
        if rec["id"] not in ids:
            continue
        if rec["id"] in seen:
            raise ValueError(f"question {rec['id']} appears twice in the poisons file")
        seen.add(rec["id"])
        found = set()
        for p in POISON_TYPES:
            for i, clean in enumerate(rec["clean"]):
                res = check_poison(clean, rec["poisoned"][p.name][i], rec["poisoned"]["swap"][i],
                                   rec["gold_answer"], rec["target_answer"], p.name, p.templates)
                found.update(res["partial"])
        out.append({"question_id": rec["id"], "leak_number": "number_leak" in found,
                    "leak_word": "word_leak" in found})
    return pd.DataFrame(out)


def effective_rho(bm25_path, ids: set[str]) -> pd.DataFrame:
    """Per question and poison type: share of retrieved answer-bearing passages that are poisoned."""
    out = []
    for rec in read_jsonl_gz(bm25_path):
        if rec["id"] in ids:
            out.append({"question_id": rec["id"],
                        **{f"eff_rho_{t}": s["effective_rho"] for t, s in rec["summary"].items()}})
    return pd.DataFrame(out)


def build_table(greedy, judge, info, leaks, eff) -> pd.DataFrame:
    """Join responses, judge labels and question tables; every join is checked."""
    df = greedy
    for task, allowed in (("flag", set(FLAG_VALUES)), ("verdict", VERDICT_LABELS)):
        part = task_labels(judge, task, allowed)
        df = df.merge(part, on=KEYS, how="left", validate="one_to_one")
        df[f"{task}_judged"] = df[f"{task}_judged"].eq(True)
        if df[f"{task}_judged"].sum() != len(part):
            raise ValueError(f"{task} judgements without a greedy response")
    df["flag"] = df["flag_label"].map(FLAG_VALUES)
    for name, extra in (("info", info), ("leaks", leaks), ("effective rho", eff)):
        missing = set(df["question_id"]) - set(extra["question_id"])
        if missing:
            raise ValueError(f"{len(missing)} questions missing from the {name} table")
        df = df.merge(extra, on="question_id", how="left", validate="many_to_one")
    df["eff_rho"] = float("nan")
    eff_cols = [c for c in eff.columns if c.startswith("eff_rho_")]
    for col in eff_cols:
        mask = df["context_id"] == "bm25-" + col.removeprefix("eff_rho_")
        df.loc[mask, "eff_rho"] = df.loc[mask, col].astype(float)
    return df.drop(columns=eff_cols)


def check_main(df: pd.DataFrame, expected: dict = MAIN_COUNTS) -> dict:
    """Compare the assembled table with the counts recorded in Block E."""
    got = {"responses": len(df), "questions": df["question_id"].nunique(),
           "flags": int(df["flag_judged"].sum()),
           "flags_unparseable": int((df["flag_judged"] & df["flag"].isna()).sum()),
           "verdicts": int(df["verdict_judged"].sum()),
           "verdicts_unparseable": int((df["verdict_judged"] & df["verdict_label"].isna()).sum())}
    wrong = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    if wrong:
        raise ValueError(f"count check failed (got, expected): {wrong}")
    return got


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Assemble the greedy main-run table for Block F.")
    ap.add_argument("--run", default=str(RUN_DIR))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    run = Path(a.run)
    greedy = load_greedy(run / "_scored" / "scored.jsonl.gz")
    judge = pd.DataFrame(list(read_jsonl_gz(run / "_judge" / "judge.jsonl.gz")))
    ids = set(greedy["question_id"])
    info = question_info(PROCESSED_DIR / "contexts.jsonl.gz", PROCESSED_DIR / "questions.jsonl.gz",
                         RAW_HOTPOT, ids)
    df = build_table(greedy, judge, info,
                     leak_flags(PROCESSED_DIR / "poisons.jsonl.gz", ids),
                     effective_rho(PROCESSED_DIR / "bm25_contexts.jsonl.gz", ids))
    print("Counts:", check_main(df))
    base = df[df["context_id"] == "clean-0.00"]
    print("Flag at rho=0:", base.groupby("model")["flag"].mean().round(3).to_dict())
    print("Poisoned at rho=0:", int((base["outcome"] == "Poisoned").sum()))
    q = df.drop_duplicates("question_id")
    print("HotpotQA type:", q["hotpot_type"].value_counts().to_dict())
    print("Leak questions:", {"number": int(q["leak_number"].sum()), "word": int(q["leak_word"].sum())})
    bm = df[df["context_id"].str.startswith("bm25-")]
    print("Mean effective rho:", bm.groupby("context_id")["eff_rho"].mean().round(3).to_dict())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print("Written", out, len(df), "rows")


if __name__ == "__main__":
    main()