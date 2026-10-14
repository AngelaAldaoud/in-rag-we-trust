"""Self-consistency macros (Block F5 addendum): results/main/sc.csv -> results/analysis/numbers_sc.tex.
Same rules as numbers.py (letters-only names, unique, generated, D277-D278); all names start with Sc (D280).
Run with:
    uv run python -m inragwetrust.analysis.sc_numbers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from inragwetrust.analysis.figures import need
from inragwetrust.analysis.numbers import MODEL, PROMPT, Macros, pct, rho_name, word

SC = Path("results/main/sc.csv")
OUT = Path("results/analysis/numbers_sc.tex")


def level_name(v) -> str:
    try:
        return rho_name(v)
    except (TypeError, ValueError, KeyError):
        return word(v)


def build(sc: pd.DataFrame) -> Macros:
    need(sc, ["model", "prompt", "level", "sc", "sc_outcome", "questions"], "sc")
    M = Macros()
    M.section("Self-consistency from results/main/sc.csv (percent); ScN = questions")
    for _, r in sc.iterrows():
        base = MODEL[r["model"]] + PROMPT[r["prompt"]] + level_name(r["level"])
        M.add("Sc" + base, pct(r["sc"]))
        M.add("ScOutcome" + base, pct(r["sc_outcome"]))
        M.add("ScN" + base, str(int(r["questions"])))
    rho = sc[pd.to_numeric(sc["level"], errors="coerce").notna()]
    M.add("ScRangeMin", pct(rho["sc"].min()))
    M.add("ScRangeMax", pct(rho["sc"].max()))
    M.add("ScOutcomeRangeMin", pct(rho["sc_outcome"].min()))
    M.add("ScOutcomeRangeMax", pct(rho["sc_outcome"].max()))
    return M


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sc", default=str(SC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    M = build(pd.read_csv(a.sc))
    Path(a.out).write_text(M.tex(), encoding="utf-8")
    print(f"Written {a.out}: {len(M.items)} macros")
    for k in ("ScLlamaConsRhoZero", "ScOutcomeLlamaConsRhoZero", "ScRangeMin", "ScRangeMax"):
        print(f"  \\{k} = {M.items[k]}")


if __name__ == "__main__":
    main()