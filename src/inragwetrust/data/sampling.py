"""Pilot / main / reserve split of the eligible pool (D57, D60, D92, D96)."""

from __future__ import annotations

import random

N_PILOT = 50
N_MAIN = 500


def split_ids(
    ids: list[str], seed: int, n_pilot: int = N_PILOT, n_main: int = N_MAIN
) -> dict[str, tuple[str, int]]:
    """Shuffle the ids once and cut them into pilot, main and reserve.

    Returns id -> (split, rank). The rank is the position in the shuffled
    order; failed questions are replaced by reserve questions in rank order.
    """
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question ids.")
    if len(ids) < n_pilot + n_main:
        raise ValueError(f"Need at least {n_pilot + n_main} questions, got {len(ids)}.")
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    result: dict[str, tuple[str, int]] = {}
    for rank, qid in enumerate(order):
        if rank < n_pilot:
            split = "pilot"
        elif rank < n_pilot + n_main:
            split = "main"
        else:
            split = "reserve"
        result[qid] = (split, rank)
    return result 