from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inragwetrust.analysis import numbers as nb

REAL = Path("results/analysis")


def test_macro_names_letters_only_and_unique():
    M = nb.Macros()
    M.add("AsrLlama", "1.0")
    with pytest.raises(ValueError, match="letters only"):
        M.add("Asr1", "1.0")
    with pytest.raises(ValueError, match="duplicate"):
        M.add("AsrLlama", "2.0")
    assert "\\newcommand{\\AsrLlama}{1.0}" in M.tex()


def test_formatting_and_names():
    assert nb.num(-2.4) == r"\ensuremath{-}2.4" and nb.num(-0.0004) == "0.0" and nb.num(np.nan) == "--"
    assert nb.pct(0.5533) == "55.3" and nb.pval(1e-9) == r"\ensuremath{<}0.001" and nb.pval(0.0123) == "0.012"
    assert nb.rho_name(0.75) == "RhoThreeQuarters" and nb.group_name("rho=0.25") == "RhoQuarter"
    assert nb.group_name("closed-book") == "ClosedBook" and nb.word("rho:type") == "RhoType"
    assert nb.yes_no(True) == "Yes" and nb.yes_no("False") == "No"


def test_effective_rho_bins_get_distinct_names():
    assert nb.eff_name("none") == "EffNone" and nb.eff_name(np.nan) == "EffNone" and nb.eff_name(None) == "EffNone"
    assert nb.eff_name("0.00") == "EffZero" and nb.eff_name("0.50") == "EffFifty" and nb.eff_name(1.0) == "EffHundred"
    assert nb.eff_name("0.33") == "EffThirtyThree" and nb.eff_name("0.67") == "EffSixtySeven"
    names = {nb.eff_name(x / 100) for x in range(101)}
    assert len(names) == 101
    with pytest.raises(ValueError):
        nb.number_word(101)


def test_largest_step_and_ranges():
    rhos, asr = [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.05, 0.15, 0.32, 0.55]
    d = pd.DataFrame([{"model": "llama", "prompt": "all", "rho": r, "asr": a, "asr_lo": a, "asr_hi": a,
                       "acc": 0.6 - a, "acc_lo": 0.5 - a, "acc_hi": 0.7 - a, "flag": 0.3, "flag_lo": 0.2,
                       "flag_hi": 0.4} for r, a in zip(rhos, asr)])
    M = nb.Macros()
    nb.descriptives(M, d)
    assert M.items["StepMaxLlama"] == "23.0"
    assert (M.items["StepFromLlama"], M.items["StepToLlama"]) == ("0.75", "1.00")
    assert M.items["AsrRangeRhoOneMax"] == "55.0" and M.items["AsrLlamaAllRhoOneLo"] == "55.0"


def test_leak_question_count_must_agree_between_models():
    rows = [{"model": m, "leak_any": leak, "n_questions": (70 if m == "llama" else 71) if leak else 430,
             "poisoned": 0.2, "poisoned_lo": 0.1, "poisoned_hi": 0.3}
            for m in ("llama", "qwen") for leak in (True, False)]
    with pytest.raises(ValueError, match="leak-question count"):
        nb.leaks(nb.Macros(), pd.DataFrame(rows))


def test_transitions_other_and_mdd_counts():
    tr = pd.DataFrame([{"model": "llama", "meta": "consistency", "direction": "plain Poisoned -> meta",
                        "other_outcome": o, "share": s}
                       for o, s in (("Poisoned", 0.67), ("Abstain", 0.11), ("Correct", 0.10), ("NoAnswer", 0.12))])
    M = nb.Macros()
    nb.transitions(M, tr)
    assert M.items["TransLlamaConsOther"] == "12.0" and M.items["TransStayMax"] == "67.0"
    md = pd.DataFrame({"mdd": [0.04, 0.05, 0.06, np.nan], "power_5pt": [0.95, 0.7, 0.5, np.nan]})
    nb.mdd(M, md)
    assert M.items["MddMedian"] == "5.0" and M.items["MddUnderpowered"] == "2" and M.items["MddNoPower"] == "1"


@pytest.mark.skipif(not (REAL / "descriptive_by_rho.csv").exists(), reason="frozen results not present")
def test_frozen_numbers_regression():
    M = nb.build(REAL)
    expected = {"NQuestions": "500", "FOneOrLlama": "53.7", "FOneOrMistral": "64.9", "FOneOrQwen": "50.3",
                "CrossRulesLlama": "0.75", "CrossUpperLlama": "1.00", "FTwoAsrSig": "5", "FTwoAsrTests": "6",
                "FThreeSig": "3", "FFourSig": "4", "MddCells": "96", "AuditFlagged": "9", "AuditCandidates": "73",
                "SensConclusions": "28", "SensFlaggedSame": "28", "SensLeaksSame": "26"}
    assert {k: M.items[k] for k in expected} == expected