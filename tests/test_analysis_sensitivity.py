import numpy as np
import pandas as pd
import pytest

from inragwetrust.analysis import sensitivity as sv


def test_conclusion_directions():
    assert sv.conclusion(1.5, 0.01, 1.0) == "up" and sv.conclusion(0.8, 0.01, 1.0) == "down"
    assert sv.conclusion(-0.02, 0.2, 0.0) == "n.s." and sv.conclusion(20.0, 0.001, np.nan) == "sig"
    assert sv.conclusion(np.nan, np.nan, 1.0) == "n.s."


def test_compare_marks_changed_conclusions():
    primary = pd.DataFrame({"model": ["a", "b"], "estimate": [0.8, 0.9], "ci_low": [0.7, 0.8],
                            "ci_high": [0.9, 0.99], "p_holm": [0.001, 0.03]})
    sens = primary.assign(p_holm=[0.002, 0.08])
    out = sv.compare(primary, sens, ["model"], 1.0).set_index("model")
    assert out.loc["a", "same_conclusion"] and not out.loc["b", "same_conclusion"]
    assert out.loc["b", "conclusion_sens"] == "n.s."


def test_excluded_ids(tmp_path):
    df = pd.DataFrame({"question_id": ["q1", "q1", "q2", "q3"], "leak_word": [True, True, False, False],
                       "leak_number": [False, False, False, True]})
    assert sv.excluded_ids("leaks", df, tmp_path) == {"q1", "q3"}
    with pytest.raises(FileNotFoundError, match="audit flagged"):
        sv.excluded_ids("flagged", df, tmp_path)
    pd.DataFrame({"question_id": ["q2"]}).to_csv(tmp_path / "flagged_questions.csv", index=False)
    assert sv.excluded_ids("flagged", df, tmp_path) == {"q2"}