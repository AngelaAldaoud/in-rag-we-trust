"""Exploratory analyses for Block F3 (analysis plan section 5). Intervals only, no significance claims.
Comparisons between groups of questions use question-clustered bootstrap intervals (D264).
Run with:
    uv run python -m inragwetrust.analysis.exploratory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from inragwetrust.analysis.confirmatory import PLAIN, controlled
from inragwetrust.analysis.data import OUT as TABLE
from inragwetrust.analysis.stats import BOOTSTRAP, SEED, cluster_bootstrap, wilson
from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.evaluation.metrics import response_em_f1, upper_outcome
from inragwetrust.poisoning.build import read_jsonl_gz

RESULTS = Path("results/analysis")
NO_USABLE = ("Abstain", "NoAnswer")
PRECISION, MISS_RATE = "9/13", "0/18"
D127_FIRST_SLOT0 = 102


def rate(k, n) -> tuple[float, float, float]:
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = wilson(int(k), int(n))
    return k / n, lo, hi


def rates(d: pd.DataFrame, groups: list[str], cols: list[str]) -> pd.DataFrame:
    """Rates of 0/1 columns per condition group, with Wilson 95% intervals (plan section 3)."""
    rows = []
    for key, g in d.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key), n=len(g))
        for col in cols:
            v = g[col].dropna()
            row[col], row[col + "_lo"], row[col + "_hi"] = rate(v.sum(), len(v))
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_rates(d: pd.DataFrame, groups: list[str], cols: list[str]) -> pd.DataFrame:
    """Rates per group of questions with question-clustered bootstrap 95% intervals (D264)."""
    rows = []
    for key, g in d.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key), n=len(g), n_questions=g["question_id"].nunique())
        for col in cols:
            per_q = g.groupby("question_id")[col].agg(["sum", "count"])
            row[col] = per_q["sum"].sum() / per_q["count"].sum()
            row[col + "_lo"], row[col + "_hi"] = cluster_bootstrap(per_q["sum"], per_q["count"])
        rows.append(row)
    return pd.DataFrame(rows)


def context_group(cid: str) -> str:
    if cid == "closedbook":
        return "closed-book"
    if cid.startswith("bm25-"):
        return "retrieval"
    return f"rho={float(cid.rsplit('-', 1)[1]):.2f}"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, labels in (("poisoned", ("Poisoned",)), ("correct", ("Correct",)), ("abstain", ("Abstain",)),
                        ("nousable", NO_USABLE)):
        df[col] = df["outcome"].isin(labels).astype(int)
    df["group"] = df["context_id"].map({c: context_group(c) for c in df["context_id"].unique()})
    return df


def add_upper(df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Judge-based upper-bound outcomes with the frozen upper_outcome (D170, D175)."""
    up = pd.Series([upper_outcome(r, None if pd.isna(r["verdict_label"]) else r["verdict_label"])
                    for r in records], index=df.index)
    return df.assign(correct_plus=up.eq("Correct").astype(int), poisoned_plus=up.eq("Poisoned").astype(int))


def paired_diff(pairs: pd.DataFrame, before: str, after: str) -> tuple[float, float, float]:
    diff = pairs[after] - pairs[before]
    per_q = diff.groupby(level="question_id").agg(["sum", "count"])
    lo, hi = cluster_bootstrap(per_q["sum"], per_q["count"])
    return float(diff.mean()), lo, hi


def prompt_effects(c: pd.DataFrame) -> pd.DataFrame:
    """Paired meta-prompt minus plain differences by model and rho, question-clustered bootstrap CI."""
    rows, metas = [], sorted(set(c["prompt"]) - {PLAIN})
    for (m, r), d in c.groupby(["model", "rho"]):
        for col in ("poisoned", "correct", "abstain", "nousable"):
            p = d.pivot(index=["question_id", "poison_type"], columns="prompt", values=col)
            for meta in metas:
                est, lo, hi = paired_diff(p, PLAIN, meta)
                rows.append({"model": m, "rho": r, "meta": meta, "metric": col, "n_pairs": len(p),
                             "delta": est, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def transitions(c: pd.DataFrame) -> pd.DataFrame:
    """Outcome under the other prompt when plain (or the meta-prompt) gave the poisoned answer, rho > 0."""
    rows, metas = [], sorted(set(c["prompt"]) - {PLAIN})
    for m, d in c[c["rho"] > 0].groupby("model"):
        p = d.pivot(index=["question_id", "poison_type", "rho"], columns="prompt", values="outcome")
        for meta in metas:
            for direction, counts in (("plain Poisoned -> meta", p.loc[p[PLAIN] == "Poisoned", meta].value_counts()),
                                      ("meta Poisoned <- plain", p.loc[p[meta] == "Poisoned", PLAIN].value_counts())):
                for outcome, k in counts.items():
                    rows.append({"model": m, "meta": meta, "direction": direction, "other_outcome": outcome,
                                 "count": int(k), "share": k / counts.sum()})
    return pd.DataFrame(rows)


def weights(n: int, n_boot: int = BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    """Bootstrap resamples of n clusters as counts (how often each question is drawn)."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot)


def crossover(d: pd.DataFrame, w: np.ndarray, asr: str = "poisoned", acc: str = "correct") -> dict:
    """Smallest rho with ASR > Acc (D27), with a question-bootstrap 95% interval."""
    agg = d.groupby(["question_id", "rho"])[[asr, acc]].sum()
    p, c = agg[asr].unstack(), agg[acc].unstack()
    rhos = p.columns.to_numpy()

    def first(pp, cc):
        above = pp > cc
        return np.where(above.any(-1), rhos[above.argmax(-1)], np.nan)

    point = float(first(p.to_numpy(float).sum(axis=0), c.to_numpy(float).sum(axis=0)))
    boot = first(w @ p.to_numpy(float), w @ c.to_numpy(float))
    ok = boot[~np.isnan(boot)]
    lo = float(np.percentile(ok, 2.5, method="lower")) if len(ok) else np.nan
    hi = float(np.percentile(ok, 97.5, method="higher")) if len(ok) else np.nan
    return {"crossover": point, "lo": lo, "hi": hi, "share_no_crossover": float(np.isnan(boot).mean())}


def crossovers(c: pd.DataFrame) -> pd.DataFrame:
    """Rule-based crossover per model and per curve; upper-bound crossover per model (D265)."""
    w = weights(c["question_id"].nunique())
    rows = []
    for m in sorted(c["model"].unique()):
        d = c[c["model"] == m]
        rows.append({"model": m, "prompt": "all", "poison_type": "all", "scoring": "rules", **crossover(d, w)})
        rows.append({"model": m, "prompt": "all", "poison_type": "all", "scoring": "upper bound",
                     **crossover(d, w, "poisoned_plus", "correct_plus")})
    for (m, p, t), _ in c[c["rho"] > 0].groupby(["model", "prompt", "poison_type"]):
        d = c[(c["model"] == m) & (c["prompt"] == p) & ((c["poison_type"] == t) | (c["rho"] == 0))]
        rows.append({"model": m, "prompt": p, "poison_type": t, "scoring": "rules", **crossover(d, w)})
    return pd.DataFrame(rows)


def flag_vs_baseline(c: pd.DataFrame) -> pd.DataFrame:
    """Flag rate minus the same questions' rho = 0 flag rate, with a question bootstrap CI."""
    rows = []
    for m, d in c.groupby("model"):
        per_q = d.groupby(["question_id", "rho"])["flag"].mean().unstack()
        for r in per_q.columns:
            diff = per_q[r] - per_q[0.0]
            lo, hi = cluster_bootstrap(diff, np.ones(len(diff)))
            rows.append({"model": m, "rho": r, "flag": d.loc[d["rho"] == r, "flag"].mean(),
                         "baseline": d.loc[d["rho"] == 0, "flag"].mean(), "delta": diff.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def flag_cells(c: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "prompt", "poison_type", "rho"]
    pos = c[c["rho"] > 0]
    f = rates(pos[pos["flag"] == 1], keys, ["poisoned"]).add_suffix("_flagged")
    u = rates(pos[pos["flag"] == 0], keys, ["poisoned"]).add_suffix("_unflagged")
    return f.merge(u, left_on=[k + "_flagged" for k in keys], right_on=[k + "_unflagged" for k in keys])


def bracket(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based Acc/ASR next to the judge-based upper bound Acc+/ASR+."""
    return rates(df, ["model", "group"], ["correct", "correct_plus", "poisoned", "poisoned_plus"])


def no_answer(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, d in df.groupby("model"):
        na = d["outcome"].eq("NoAnswer")
        trunc = d["truncated"].astype(bool)
        rows.append({"model": m, "n": len(d), "no_answer": na.mean(), "no_answer_truncated": (na & trunc).mean(),
                     "no_answer_stopped": (na & ~trunc).mean(), "cap_hits_all": trunc.mean()})
    return pd.DataFrame(rows)


def hallucination(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(hall_any=df["hall"].eq(True).astype(int), hall_strict_any=df["hall_strict"].eq(True).astype(int))
    over_all = rates(d, ["model", "group"], ["hall_any", "hall_strict_any"])
    ow = rates(d[d["outcome"] == "OtherWrong"], ["model", "group"], ["hall_any"])
    ow = ow.rename(columns={c: "ow_" + c for c in ow.columns if c not in ("model", "group")})
    return over_all.merge(ow, on=["model", "group"], how="left").assign(precision=PRECISION, miss_rate=MISS_RATE)


def em_f1(df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    vals = np.array([response_em_f1(r, r["gold_answer"]) for r in records], dtype=float)
    d = df.assign(em=vals[:, 0], f1=vals[:, 1])
    return d.groupby(["model", "group"])[["em", "f1"]].mean().reset_index()


def prior_belief(df: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """ASR at rho > 0 by what the same model answered without passages (includes H1's closed-book part)."""
    cb = df.loc[df["context_id"] == "closedbook", ["model", "question_id", "outcome"]]
    c2 = c[c["rho"] > 0].merge(cb.rename(columns={"outcome": "cb_outcome"}), on=["model", "question_id"],
                               how="left", validate="many_to_one")
    c2["cb_group"] = np.select([c2["cb_outcome"].eq("Poisoned"), c2["cb_outcome"].eq("Correct")],
                               ["target without passages", "gold without passages"], "other")
    return cluster_rates(c2, ["model", "cb_group", "rho"], ["poisoned", "correct"])


def retrieval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = df[df["group"] == "retrieval"].copy()
    r["poison_type"] = r["context_id"].str.removeprefix("bm25-")
    by_cond = rates(r, ["model", "prompt", "poison_type"], ["poisoned", "correct"]).merge(
        r.groupby(["model", "prompt", "poison_type"])["eff_rho"].mean().rename("mean_eff_rho").reset_index())
    r["eff_bin"] = r["eff_rho"].map(lambda x: "none" if pd.isna(x) else f"{x:.2f}")
    return by_cond, cluster_rates(r, ["model", "eff_bin"], ["poisoned", "correct"])


def slot_position(layout: dict, slot: int) -> int:
    """Display position (0-4) of an answer-bearing slot, with the bridge inserted (D123, D124)."""
    items = list(layout["display_order"])
    items.insert(layout["bridge_pos"], "bridge")
    return items.index(slot)


def first_poison(ids: set[str]) -> pd.DataFrame:
    rows = []
    for rec in read_jsonl_gz(PROCESSED_DIR / "contexts.jsonl.gz"):
        if rec["id"] in ids:
            slot = rec["layout"]["poison_order"][0]
            rows.append({"question_id": rec["id"], "first_slot": slot,
                         "first_position": slot_position(rec["layout"], slot)})
    out = pd.DataFrame(rows)
    n0 = int(out["first_slot"].eq(0).sum())
    if n0 != D127_FIRST_SLOT0:
        raise ValueError(f"slot 0 poisoned first in {n0} questions; D127 recorded {D127_FIRST_SLOT0}")
    return out


def position_and_leaks(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    one = c[c["rho"] == 0.25].merge(first_poison(set(c["question_id"])), on="question_id", validate="many_to_one")
    one["first_is_original"] = one["first_slot"].eq(0)
    pos = c[c["rho"] > 0].assign(leak_any=lambda t: t["leak_word"] | t["leak_number"])
    return (cluster_rates(one, ["model", "first_position"], ["poisoned"]),
            cluster_rates(one, ["model", "first_is_original"], ["poisoned"]),
            cluster_rates(pos, ["model", "leak_any"], ["poisoned"]))


def mdd(p_d: float, n: int = 500, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest difference detectable with the given power (Block A McNemar formula)."""
    if not 0 < p_d < 1:
        return np.nan
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return float(brentq(lambda d: (za * np.sqrt(p_d) + zb * np.sqrt(p_d - d * d)) ** 2 / d ** 2 - n,
                        1e-6, np.sqrt(p_d) - 1e-12))


def power_at(p_d: float, delta: float, n: int = 500, alpha: float = 0.05) -> float:
    if p_d <= delta ** 2:
        return np.nan
    za = norm.ppf(1 - alpha / 2)
    return float(norm.cdf((np.sqrt(n) * delta - za * np.sqrt(p_d)) / np.sqrt(p_d - delta ** 2)))


def mdd_cells(c: pd.DataFrame) -> pd.DataFrame:
    rows, metas = [], sorted(set(c["prompt"]) - {PLAIN})
    for (m, t, r), d in c[c["rho"] > 0].groupby(["model", "poison_type", "rho"]):
        p = d.pivot(index="question_id", columns="prompt", values="poisoned")
        for meta in metas:
            p_d = float((p[PLAIN] != p[meta]).mean())
            rows.append({"model": m, "poison_type": t, "rho": r, "meta": meta, "n": len(p), "p_d": p_d,
                         "mdd": mdd(p_d, len(p)), "power_5pt": power_at(p_d, 0.05, len(p))})
    return pd.DataFrame(rows)


def show(title: str, t: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(t.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Block F3 exploratory analyses.")
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    df = add_indicators(pd.read_parquet(a.table))
    records = df.to_dict("records")
    df = add_upper(df, records)
    c = controlled(df)
    c["nousable"] = c["outcome"].isin(NO_USABLE).astype(int)
    by_cond, by_eff = retrieval(df)
    position, original, leaks = position_and_leaks(c)
    t = {"prompt_effects": prompt_effects(c), "transitions": transitions(c), "crossover": crossovers(c),
         "flag_baseline": flag_vs_baseline(c), "flag_cells": flag_cells(c), "bracket": bracket(df),
         "no_answer": no_answer(df), "hallucination": hallucination(df), "em_f1": em_f1(df, records),
         "closed_book": rates(df[df["group"] == "closed-book"], ["model"], ["correct", "poisoned", "abstain", "nousable"]),
         "prior_belief": prior_belief(df, c), "retrieval_conditions": by_cond, "retrieval_eff_rho": by_eff,
         "position": position, "original_first": original, "leaks": leaks, "mdd": mdd_cells(c),
         "type_model": rates(c[c["rho"] > 0], ["model", "poison_type", "rho"], ["poisoned"])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "explore_slot0.csv"
    if stale.exists():
        stale.unlink()
    for name, table in t.items():
        table.to_csv(out / f"explore_{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        cr = t["crossover"]
        show("2 Crossover per model: rules vs upper bound", cr[cr["prompt"] == "all"])
        show("2 Curves not crossing at 0.75", cr[(cr["prompt"] != "all") & (cr["crossover"] != 0.75)])
        cols = ["model", "cb_group", "rho", "n_questions", "poisoned", "poisoned_lo", "poisoned_hi"]
        show("8b Prior belief, clustered CI", t["prior_belief"][cols])
        show("8c Retrieval by effective rho, clustered CI",
             t["retrieval_eff_rho"][["model", "eff_bin", "n_questions", "poisoned", "poisoned_lo", "poisoned_hi"]])
        keep = ["n_questions", "poisoned", "poisoned_lo", "poisoned_hi"]
        show("9a Position at rho=0.25, clustered CI", t["position"][["model", "first_position"] + keep])
        show("9b Original passage poisoned first (D127: 102 questions), clustered CI",
             t["original_first"][["model", "first_is_original"] + keep])
        show("9c Leaks, clustered CI", t["leaks"][["model", "leak_any"] + keep])
    print("\nWritten", len(t), "tables to", out)


if __name__ == "__main__":
    main()