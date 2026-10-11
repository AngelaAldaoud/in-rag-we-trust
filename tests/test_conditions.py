"""The condition table matches the A3 design (C1, D125)."""

from collections import Counter

from inragwetrust.conditions import MODELS, PROMPTS, condition_table, context_ids


def test_size_matches_design():
    table = condition_table()
    assert len(table) == 192
    assert Counter(c.run for c in table) == Counter(
        poisoned=144, baseline=9, closed_book=3, retrieval=36)


def test_ids_are_unique():
    ids = [c.condition_id for c in condition_table()]
    assert len(ids) == len(set(ids))


def test_closed_book_uses_plain_prompt_only():
    closed = [c for c in condition_table() if c.run == "closed_book"]
    assert {c.model for c in closed} == set(MODELS)
    assert {c.prompt for c in closed} == {"plain"}


def test_every_model_and_prompt_has_the_full_poison_grid():
    for model in MODELS:
        for prompt in PROMPTS:
            cells = {(c.poison_type, c.rho) for c in condition_table()
                     if c.run == "poisoned" and c.model == model and c.prompt == prompt}
            assert len(cells) == 16


def test_22_contexts_per_question():
    ids = context_ids()
    assert len(ids) == 22
    assert ids[0] == "closedbook" and "clean-0.00" in ids


def test_model_major_order():
    models = [c.model for c in condition_table()]
    switches = sum(a != b for a, b in zip(models, models[1:]))
    assert switches == len(MODELS) - 1        # each model forms one block