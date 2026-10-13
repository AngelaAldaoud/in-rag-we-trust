"""Question audit for Block F1 (D188): candidate gold errors and comparison-type questions.

Exploratory: flagged questions are documented and used only in the sensitivity analysis;
the primary analysis keeps all 500 questions (analysis plan, section 6).
    uv run python -m inragwetrust.analysis.audit sheet     # write the review sheet
    uv run python -m inragwetrust.analysis.audit flagged   # read it, write the decisions
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from inragwetrust.analysis.data import OUT as TABLE
from inragwetrust.textmatch import contains_normalized

BASELINE = ("closedbook", "clean-0.00")
SHEET = TABLE.parent / "audit_sheet.txt"
RESULTS = Path("results/analysis")
OR_PATTERN = re.compile(r"\b(?:or|either)\b", re.IGNORECASE)
SIGNALS = ("cb_poisoned", "rho0_poisoned", "baseline_verdict_t")
HEADER = """# Audit sheet (Block F1, D188). For every question fill the three lines at the bottom of its block:
#   gold_error: yes / no   yes = the HotpotQA gold is wrong AND our target is the true answer (check Wikipedia)
#   comparison: yes / no   yes = the question compares named entities and the answer is one of them
#   note: one line, required for every yes (source URL or reason)
# Do not edit any other line. Signals: cb_poisoned = closed-book answers equal to the target;
# rho0_poisoned = clean-context answers equal to the target; baseline_verdict_t = closed-book or
# clean OtherWrong answers the judge reads as the target.
"""


def gold_error_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Per question: closed-book and clean-context responses that point to the target."""
    base = df[df["context_id"].isin(BASELINE)].copy()
    cb = base["context_id"].eq("closedbook")
    poisoned = base["outcome"].eq("Poisoned")
    verdict_t = base["outcome"].eq("OtherWrong") & base["verdict_label"].eq("T")
    base["cb_poisoned"] = cb & poisoned
    base["rho0_poisoned"] = ~cb & poisoned
    base["baseline_verdict_t"] = verdict_t
    sig = base.groupby("question_id")[list(SIGNALS)].sum().astype(int)
    answers: dict[str, list[str]] = {}
    for r in base[poisoned | verdict_t].itertuples():
        text = str(r.answer).replace("\n", " ")[:120]
        answers.setdefault(r.question_id, []).append(f"{r.model}/{r.prompt}/{r.context_id}: {text}")
    sig["answers"] = pd.Series({k: " | ".join(v) for k, v in answers.items()}, dtype=object)
    sig["answers"] = sig["answers"].fillna("")
    return sig.reset_index()


def comparison_signals(info: pd.DataFrame) -> pd.DataFrame:
    """Per question: does the wording look like a comparison question?"""
    out = info[["question_id", "question", "gold_answer", "target_answer"]].copy()
    out["has_or"] = out["question"].str.contains(OR_PATTERN)
    out["gold_in_question"] = [contains_normalized(g, q) for g, q in zip(out["gold_answer"], out["question"])]
    return out


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Questions with at least one gold-error or comparison signal, with the reason."""
    sig = comparison_signals(df.drop_duplicates("question_id"))
    sig = sig.merge(gold_error_signals(df), on="question_id", how="left", validate="one_to_one")
    for c in SIGNALS:
        sig[c] = sig[c].fillna(0).astype(int)
    sig["answers"] = sig["answers"].fillna("")
    gold = sig[list(SIGNALS)].sum(axis=1) > 0
    comp = sig["has_or"] | sig["gold_in_question"]
    sig["reason"] = ["+".join(n for n, f in (("gold_error", g), ("comparison", c)) if f)
                     for g, c in zip(gold, comp)]
    return sig[gold | comp].reset_index(drop=True)


def write_sheet(cands: pd.DataFrame, path: Path, force: bool = False) -> None:
    """Write the review sheet; never overwrites existing labels unless forced."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; it may hold your labels (use --force to replace it)")
    lines = [HEADER]
    for i, r in enumerate(cands.itertuples(), 1):
        lines += [f"=== Q {i}/{len(cands)}  id={r.question_id}  reason={r.reason}",
                  f"question: {r.question}", f"gold: {r.gold_answer}", f"target: {r.target_answer}",
                  f"signals: cb_poisoned={r.cb_poisoned}, rho0_poisoned={r.rho0_poisoned}, "
                  f"baseline_verdict_t={r.baseline_verdict_t}, has_or={r.has_or}, "
                  f"gold_in_question={r.gold_in_question}",
                  f"answers: {r.answers}", "gold_error: ", "comparison: ", "note: ", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def read_sheet(path: Path) -> pd.DataFrame:
    """Read the filled sheet; every block needs yes/no twice and a note for every yes."""
    rows, problems = [], []
    for block in re.split(r"^=== Q ", path.read_text(), flags=re.M)[1:]:
        qid = re.search(r"id=(\S+)", block).group(1)
        reason = re.search(r"reason=(\S+)", block).group(1)
        vals = {}
        for f in ("gold_error", "comparison", "note"):
            m = re.search(rf"^{f}:[ \t]*(.*)$", block, flags=re.M)
            vals[f] = m.group(1).strip() if m else ""
        for f in ("gold_error", "comparison"):
            if vals[f].lower() not in ("yes", "no"):
                problems.append(f"{qid}: {f} must be yes or no")
        if "yes" in (vals["gold_error"].lower(), vals["comparison"].lower()) and not vals["note"]:
            problems.append(f"{qid}: a note is required for every yes")
        rows.append({"question_id": qid, "reason": reason,
                     "gold_error": vals["gold_error"].lower() == "yes",
                     "comparison": vals["comparison"].lower() == "yes", "note": vals["note"]})
    if problems:
        raise ValueError("sheet incomplete:\n" + "\n".join(problems))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Block F1 question audit (D188).")
    ap.add_argument("action", choices=["sheet", "flagged"])
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--sheet", default=str(SHEET))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.table)
    if a.action == "sheet":
        c = candidates(df)
        write_sheet(c, Path(a.sheet), a.force)
        print("Candidates:", len(c), c["reason"].value_counts().to_dict(), "->", a.sheet)
        for key in ("Money Trees", "Pellegrini"):
            print(f"Known case '{key}' among candidates:", bool(c["question"].str.contains(key).any()))
    else:
        dec = read_sheet(Path(a.sheet))
        info = df.drop_duplicates("question_id")[["question_id", "question", "gold_answer", "target_answer"]]
        dec = dec.merge(info, on="question_id", how="left", validate="one_to_one")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        dec.to_csv(out / "audit_decisions.csv", index=False)
        flagged = dec[dec["gold_error"] | dec["comparison"]]
        flagged.to_csv(out / "flagged_questions.csv", index=False)
        print(len(dec), "reviewed;", len(flagged), "flagged:",
              {"gold_error": int(dec["gold_error"].sum()), "comparison": int(dec["comparison"].sum())})


if __name__ == "__main__":
    main()