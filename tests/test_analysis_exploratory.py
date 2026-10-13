import numpy as np
import pandas as pd

from inragwetrust.analysis import exploratory as ex


def test_mdd_and_power_match_block_d_table():
    assert abs(ex.mdd(0.11) * 100 - 4.2) < 0.1
    assert abs(ex.mdd(0.18) * 100 - 5.3) < 0.1
    assert abs(ex.mdd(0.28) * 100 - 6.6) < 0.1
    assert abs(ex.power_at(0.11, 0.05) - 0.92) < 0.01
    assert np.isnan(ex.mdd(0.0)) and np.isnan(ex.power_at(0.002, 0.05))


def test_crossover_point_and_interval():
    rows = [{"question_id": f"q{i}", "rho": r, "poisoned": int(r >= 0.75), "correct": int(r <= 0.5)}
            for i in range(40) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    out = ex.crossover(pd.DataFrame(rows), ex.weights(40, n_boot=200))
    assert out["crossover"] == 0.75 and out["lo"] == 0.75 and out["hi"] == 0.75
    assert out["share_no_crossover"] == 0.0


def test_slot_position_inserts_bridge():
    layout = {"display_order": [1, 3, 0, 2], "bridge_pos": 2, "poison_order": [2, 0, 3, 1]}
    assert ex.slot_position(layout, 0) == 3 and ex.slot_position(layout, 1) == 0
    assert ex.slot_position(layout, 2) == 4


def test_context_group_and_indicators():
    assert ex.context_group("closedbook") == "closed-book"
    assert ex.context_group("bm25-swap") == "retrieval"
    assert ex.context_group("swap-0.25") == "rho=0.25" and ex.context_group("clean-0.00") == "rho=0.00"
    df = ex.add_indicators(pd.DataFrame({"outcome": ["Abstain", "NoAnswer", "Poisoned"],
                                         "context_id": ["clean-0.00", "closedbook", "bm25-swap"]}))
    assert df["nousable"].tolist() == [1, 1, 0] and df["poisoned"].tolist() == [0, 0, 1]


def test_transitions_counts_rescued_answers():
    rows = []
    for q, (plain, meta) in enumerate([("Poisoned", "Abstain"), ("Poisoned", "Correct"), ("Correct", "Poisoned")]):
        for prompt, outcome in (("plain", plain), ("consistency", meta)):
            rows.append({"model": "m", "question_id": f"q{q}", "poison_type": "swap", "rho": 0.5,
                         "prompt": prompt, "outcome": outcome})
    t = ex.transitions(pd.DataFrame(rows))
    rescued = t[t["direction"] == "plain Poisoned -> meta"].set_index("other_outcome")["count"]
    assert rescued.to_dict() == {"Abstain": 1, "Correct": 1}
    new = t[t["direction"] == "meta Poisoned <- plain"].set_index("other_outcome")["count"]
    assert new.to_dict() == {"Correct": 1}