"""Sensitivity analysis (exploratory; D188, D266): the confirmatory families refitted without a set of
questions, shown next to the unchanged primary results. Run with:
    uv run python -m inragwetrust.analysis.sensitivity flagged
    uv run python -m inragwetrust.analysis.sensitivity leaks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inragwetrust.analysis import confirmatory as cf
from inragwetrust.analysis.data import OUT as TABLE

RESULTS = Path("results/analysis")
FAMILIES = {"f1": (cf.family_f1, ["model"]), "f2": (cf.family_f2, ["test", "model", "meta"]),
            "f3": (cf.family_f3, ["model"]), "f4": (cf.family_f4, ["block"])}
NULL_VALUE = {"f1": 1.0, "f2": 0.0, "f3": 1.0, "f4": np.nan}


def excluded_ids(name: str, df: pd.DataFrame, results: Path = RESULTS) -> set[str]:
    """Question ids removed by a sensitivity set: hand-flagged questions (D188) or leak questions (D266)."""
    if name == "flagged":
        path = results / "flagged_questions.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: finish the audit sheet and run 'audit flagged' first")
        return set(pd.read_csv(path)["question_id"])
    if name == "leaks":
        q = df.drop_duplicates("question_id")
        return set(q.loc[q["leak_word"] | q["leak_number"], "question_id"])
    raise ValueError(f"unknown sensitivity set: {name}")


def conclusion(estimate, p_holm, null) -> str:
    """'n.s.', or the direction of a significant result ('up'/'down'; 'sig' when there is no direction)."""
    if pd.isna(p_holm) or p_holm >= 0.05:
        return "n.s."
    if pd.isna(null):
        return "sig"
    return "up" if estimate > null else "down"


def compare(primary: pd.DataFrame, sens: pd.DataFrame, keys: list[str], null) -> pd.DataFrame:
    cols = ["estimate", "ci_low", "ci_high", "p_holm"]
    p = primary[keys + [c for c in cols if c in primary]]
    s = sens[keys + [c for c in cols if c in sens]]
    out = p.merge(s, on=keys, suffixes=("_primary", "_sens"), validate="one_to_one")
    out["conclusion_primary"] = [conclusion(e, q, null) for e, q in zip(out["estimate_primary"], out["p_holm_primary"])]
    out["conclusion_sens"] = [conclusion(e, q, null) for e, q in zip(out["estimate_sens"], out["p_holm_sens"])]
    out["same_conclusion"] = out["conclusion_primary"] == out["conclusion_sens"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sensitivity analysis of F1-F4.")
    ap.add_argument("set", choices=["flagged", "leaks"])
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    out = Path(a.out)
    df = pd.read_parquet(a.table)
    drop = excluded_ids(a.set, df, out)
    c = cf.controlled(df[~df["question_id"].isin(drop)])
    print(f"Sensitivity '{a.set}': {len(drop)} questions removed, {c['question_id'].nunique()} kept")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for fam, (fn, keys) in FAMILIES.items():
            primary = pd.read_csv(out / f"confirmatory_{fam}.csv")
            table = compare(primary, fn(c), keys, NULL_VALUE[fam])
            table.to_csv(out / f"sensitivity_{a.set}_{fam}.csv", index=False)
            print(f"\n=== {fam.upper()} ===")
            print(table.round(4).to_string(index=False))
            print("Same conclusion in", int(table["same_conclusion"].sum()), "of", len(table), "tests")


if __name__ == "__main__":
    main()