"""Tests for inragwetrust.data.sampling."""

import pytest

from inragwetrust.data.sampling import split_ids

IDS = [f"q{i:04d}" for i in range(1000)]


def test_sizes_and_disjoint():
    s = split_ids(IDS, seed=7)
    splits = [v[0] for v in s.values()]
    assert splits.count("pilot") == 50
    assert splits.count("main") == 500
    assert splits.count("reserve") == 450


def test_independent_of_input_order_and_reproducible():
    assert split_ids(IDS, seed=7) == split_ids(list(reversed(IDS)), seed=7)


def test_reserve_ranks_follow_main():
    s = split_ids(IDS, seed=7)
    assert min(r for sp, r in s.values() if sp == "reserve") == 550


def test_errors():
    with pytest.raises(ValueError):
        split_ids(["a", "a"], seed=1)
    with pytest.raises(ValueError):
        split_ids(IDS[:100], seed=1)