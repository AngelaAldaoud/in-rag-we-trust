"""Confirmatory families F1-F4 (analysis plan section 4; D250-D253) and the descriptive numbers
the hypothesis verdicts need. Run with:
    uv run python -m inragwetrust.analysis.confirmatory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inragwetrust.analysis.data import OUT as TABLE
from inragwetrust.analysis.stats import (BOOTSTRAP, SEED, cluster_bootstrap, fit_gee, holm,
                                         mcnemar_exact, odds_ratio, wilson)

RESULTS = Path("results/analysis")
PLAIN = "plain"
PROMPT = "C(prompt, Treatment('plain'))"
FORMULAS = {
    "F1": f"poisoned ~ rho + {PROMPT}",
    "F2_sens": "poisoned ~ meta + rho + C(poison_type)",
    "F3": f"poisoned ~ flag + rho + {PROMPT} + C(poison_type)",
    "F4": (f"poisoned ~ rho + {PROMPT} + C(poison_type) + C(model) + rho:C(poison_type)"
           f" + {PROMPT}:C(poison_type) + rho:C(model) + {PROMPT}:C(model)"),
}
BLOCKS = {"rho:type": ("rho", "poison_type", 3), "prompt:type": ("prompt", "poison_type", 6),
          "rho:model": ("rho", "model", 2), "prompt:model": ("prompt", "model", 4)}
NAN4 = (np.nan,) * 4


def controlled(df: pd.DataFrame) -> pd.DataFrame:
    """Controlled-context responses with numeric rho, poison type ('clean' at rho=0) and 0/1 outcomes."""
    c = df[(df["context_id"] != "closedbook") & ~df["context_id"].str.startswith("bm25-")].copy()
    parts = c["context_id"].str.rsplit("-", n=1, expand=True)
    rho = parts[1].astype(float)
    known = c["rho"].notna()
    if known.any() and not np.allclose(rho[known], c.loc[known, "rho"].astype(float)):
        raise ValueError("rho in the scored file disagrees with the context id")
    c["rho"] = rho
    c["poison_type"] = parts[0].where(rho > 0, "clean")
    for col, label in (("poisoned", "Poisoned"), ("correct", "Correct"), ("abstain", "Abstain")):
        c[col] = c["outcome"].eq(label).astype(int)
    return c


def paired_stats(pairs: pd.DataFrame, before: str, after: str) -> dict:
    """Exact McNemar on 0/1 pairs, mean difference (after - before) and question-clustered bootstrap CI."""
    if pairs[[before, after]].isna().any().any():
        raise ValueError("incomplete pairs")
    x, y = pairs[before].to_numpy(), pairs[after].to_numpy()
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    per_q = pd.Series(y - x, index=pairs.index).groupby(level="question_id").agg(["sum", "count"])
    lo, hi = cluster_bootstrap(per_q["sum"], per_q["count"])
    return {"estimate": float((y - x).mean()), "ci_low": lo, "ci_high": hi,
            "p_raw": mcnemar_exact(b, c), "n": len(x), "b": b, "c": c}


def gee_columns(fit, term: str, prefix: str = "") -> dict:
    res, cov, note = fit
    est, lo, hi, p = odds_ratio(res, term) if res is not None else NAN4
    keys = ("estimate", "ci_low", "ci_high", "p_raw", "cov_struct", "note")
    return {prefix + k: v for k, v in zip(keys, (est, lo, hi, p, cov, note))}


def with_holm(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"])
    return out


def family_f1(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in sorted(c["model"].unique()):
        d = c[c["model"] == m]
        rows.append({"family": "F1", "model": m, "estimate_kind": "OR rho 0->1", "n": len(d),
                     **gee_columns(fit_gee(FORMULAS["F1"], d), "rho")})
    return with_holm(rows)


def family_f2(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pos, zero = c[c["rho"] > 0], c[c["rho"] == 0]
    metas = sorted(set(c["prompt"]) - {PLAIN})
    for m in sorted(c["model"].unique()):
        for meta in metas:
            pa = pos[pos["model"] == m].pivot(index=["question_id", "poison_type", "rho"],
                                              columns="prompt", values="poisoned")
            sens = pos[(pos["model"] == m) & pos["prompt"].isin([PLAIN, meta])].copy()
            sens["meta"] = (sens["prompt"] == meta).astype(int)
            rows.append({"family": "F2", "test": "a", "model": m, "meta": meta,
                         "estimate_kind": "delta ASR, rho>0", **paired_stats(pa, PLAIN, meta),
                         **gee_columns(fit_gee(FORMULAS["F2_sens"], sens), "meta", "sens_")})
            for test, col, kind in (("b", "correct", "delta Acc, rho=0"), ("c", "abstain", "delta Abst, rho=0")):
                pz = zero[zero["model"] == m].pivot(index="question_id", columns="prompt", values=col)
                rows.append({"family": "F2", "test": test, "model": m, "meta": meta,
                             "estimate_kind": kind, **paired_stats(pz, PLAIN, meta)})
    out = with_holm(rows)
    nq = c["question_id"].nunique()
    expected = {"a": 16 * nq, "b": nq, "c": nq}
    if not all(out.loc[out["test"] == t, "n"].eq(n).all() for t, n in expected.items()):
        raise ValueError("unexpected number of pairs in F2")
    return out


def family_f3(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pos = c[(c["rho"] > 0) & c["flag"].notna()]
    for m in sorted(pos["model"].unique()):
        d = pos[pos["model"] == m]
        rows.append({"family": "F3", "model": m, "estimate_kind": "OR flag", "n": len(d),
                     "n_flagged": int(d["flag"].sum()),
                     "asr_flagged": d.loc[d["flag"] == 1, "poisoned"].mean(),
                     "asr_unflagged": d.loc[d["flag"] == 0, "poisoned"].mean(),
                     **gee_columns(fit_gee(FORMULAS["F3"], d), "flag")})
    return with_holm(rows)


def block_terms(names) -> dict[str, list[str]]:
    """Coefficient names of each interaction block, checked against the expected counts."""
    out = {}
    for block, (a, b, size) in BLOCKS.items():
        terms = [n for n in names if ":" in n and a in n and b in n]
        if len(terms) != size:
            raise ValueError(f"{block}: expected {size} terms, found {len(terms)}")
        out[block] = terms
    return out


def family_f4(c: pd.DataFrame) -> pd.DataFrame:
    d = c[c["rho"] > 0]
    res, cov, note = fit_gee(FORMULAS["F4"], d)
    rows = []
    for block, terms in (block_terms(res.params.index) if res is not None else {b: [] for b in BLOCKS}).items():
        stat, p = np.nan, np.nan
        if res is not None:
            r = np.zeros((len(terms), len(res.params)))
            for i, t in enumerate(terms):
                r[i, res.params.index.get_loc(t)] = 1.0
            w = res.wald_test(r, use_f=False, scalar=True)
            stat, p = float(np.squeeze(w.statistic)), float(w.pvalue)
        rows.append({"family": "F4", "block": block, "estimate_kind": "Wald chi2", "df": BLOCKS[block][2],
                     "estimate": stat, "p_raw": p, "n": len(d), "cov_struct": cov, "note": note})
    return with_holm(rows)


def descriptives(c: pd.DataFrame) -> pd.DataFrame:
    """ASR, Acc and Flag by model x prompt (and all prompts) x rho, with Wilson 95% intervals."""
    both = pd.concat([c, c.assign(prompt="all")])
    rows = []
    for (m, p, r), d in both.groupby(["model", "prompt", "rho"]):
        row = {"model": m, "prompt": p, "rho": r, "n": len(d)}
        for name, col in (("asr", "poisoned"), ("acc", "correct")):
            k = int(d[col].sum())
            row[name], (row[name + "_lo"], row[name + "_hi"]) = k / len(d), wilson(k, len(d))
        f = d["flag"].dropna()
        row["flag"], (row["flag_lo"], row["flag_hi"]) = f.mean(), wilson(int(f.sum()), len(f))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Confirmatory families F1-F4 and descriptives.")
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    c = controlled(pd.read_parquet(a.table))
    if len(c) != 76_500:
        raise ValueError(f"expected 76,500 controlled responses, found {len(c)}")
    print(f"Controlled responses: {len(c)} | bootstrap {BOOTSTRAP} resamples, seed {SEED}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"confirmatory_f1": family_f1(c), "confirmatory_f2": family_f2(c),
              "confirmatory_f3": family_f3(c), "confirmatory_f4": family_f4(c),
              "descriptive_by_rho": descriptives(c)}
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 4):
        for name, t in tables.items():
            t.to_csv(out / f"{name}.csv", index=False)
            print(f"\n=== {name} ===")
            if name == "descriptive_by_rho":
                print(t[t["prompt"] == "all"][["model", "rho", "n", "asr", "acc", "flag"]].to_string(index=False))
            else:
                print(t.drop(columns=[x for x in ("family", "note", "sens_note") if x in t]).to_string(index=False))
                notes = [n for col in ("note", "sens_note") if col in t for n in t[col].dropna() if n]
                print("Fit notes:", notes or "none")


if __name__ == "__main__":
    main()