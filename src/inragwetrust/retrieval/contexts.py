"""Context builder for the controlled-retrieval conditions (C1, D122-D124).

A context is what a model reads for one question in one condition: the 4
answer-bearing slots (each one either its clean passage or that passage's
poisoned twin) plus the bridge passage, in an order fixed per question.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from inragwetrust.data.build import SEED
from inragwetrust.poisoning.poisons import POISON_TYPES

K = 4                                   # answer-bearing slots per context
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)      # poison ratios of the design
BRIDGE = "bridge"
TYPE_NAMES = tuple(p.name for p in POISON_TYPES)   # "swap", "correct", "authority", "recency"


@dataclass(frozen=True)
class Layout:
    """Random choices for one question, drawn once and shared by every condition."""

    poison_order: tuple[int, ...]       # slots in the order they become poisoned (D122)
    display_order: tuple[int, ...]      # slots in the order they appear (D123)
    bridge_pos: int                     # where the bridge passage is inserted, 0..K (D124)

    def items(self) -> tuple:
        """The K + 1 positions of the context: slot numbers and the bridge."""
        order = list(self.display_order)
        order.insert(self.bridge_pos, BRIDGE)
        return tuple(order)


@dataclass(frozen=True)
class Passage:
    """One passage of a context, with the labels the analysis needs."""

    position: int                       # 0..K, place in the context
    role: str                           # "bridge", "clean" or "poisoned"
    slot: int | None                    # 0..3 for answer-bearing passages, None for the bridge
    text: str


def layout_for(qid: str, seed: int = SEED) -> Layout:
    """Draw the poison order, display order and bridge position of one question."""
    poison_order = list(range(K))
    random.Random(f"{seed}-{qid}-poison-order").shuffle(poison_order)

    rng = random.Random(f"{seed}-{qid}-display-order")
    display_order = list(range(K))
    rng.shuffle(display_order)
    bridge_pos = rng.randrange(K + 1)

    return Layout(tuple(poison_order), tuple(display_order), bridge_pos)


def n_poisoned(rho: float) -> int:
    """Number of poisoned slots at ratio rho (rho * K, which must be a whole number)."""
    if rho not in RHOS:
        raise ValueError(f"rho must be one of {RHOS}, got {rho}")
    return round(rho * K)


def poisoned_slots(layout: Layout, rho: float) -> frozenset[int]:
    """The first m slots of the poison order (nested across rho, D122)."""
    return frozenset(layout.poison_order[: n_poisoned(rho)])


def build_context(record: dict, poison_type: str | None, rho: float,
                  layout: Layout | None = None) -> list[Passage]:
    """The ordered passages for one question at (poison_type, rho).

    rho = 0 is the shared clean baseline and takes poison_type=None.
    """
    if rho == 0 and poison_type is not None:
        raise ValueError("the rho = 0 baseline is shared: use poison_type=None")
    if rho > 0 and poison_type not in TYPE_NAMES:
        raise ValueError(f"unknown poison type: {poison_type}")

    layout = layout or layout_for(record["id"])
    bad = poisoned_slots(layout, rho)

    context = []
    for position, item in enumerate(layout.items()):
        if item == BRIDGE:
            context.append(Passage(position, "bridge", None, record["bridge_passage"]))
        elif item in bad:
            context.append(Passage(position, "poisoned", item,
                                   record["poisoned"][poison_type][item]))
        else:
            context.append(Passage(position, "clean", item, record["clean"][item]))
    return context


def closed_book_context(record: dict) -> list[Passage]:
    """The closed-book condition: no passages at all."""
    return []