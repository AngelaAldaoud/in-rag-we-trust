from pathlib import Path

import pandas as pd
import pytest

from inragwetrust.analysis import sc_numbers as sn

REAL = Path("results/main/sc.csv")


def test_names_values_and_non_numeric_levels():
    sc = pd.DataFrame([{"model": "llama", "prompt": "plain", "level": lv, "sc": s, "sc_outcome": o, "questions": n}
                       for lv, s, o, n in (("0.00", 0.7584, 0.8596, 500), ("1.00", 0.60, 0.70, 2000),
                                           ("closed-book", 0.50, 0.55, 500))])
    M = sn.build(sc)
    assert M.items["ScLlamaPlainRhoZero"] == "75.8" and M.items["ScOutcomeLlamaPlainRhoZero"] == "86.0"
    assert M.items["ScLlamaPlainClosedBook"] == "50.0" and M.items["ScNLlamaPlainRhoOne"] == "2000"
    assert M.items["ScRangeMin"] == "60.0" and M.items["ScRangeMax"] == "75.8"


@pytest.mark.skipif(not REAL.exists(), reason="frozen sc.csv not present")
def test_frozen_sc_regression():
    M = sn.build(pd.read_csv(REAL))
    assert M.items["ScLlamaConsRhoZero"] == "75.8" and M.items["ScOutcomeLlamaConsRhoZero"] == "86.0"
    assert len(M.items) == 3 * 57 + 4