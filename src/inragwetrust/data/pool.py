"""The eligible question pool: the filters of the Data section.

Filters, in order:
  (i)   the gold answer is not yes/no                         (D29)
  (ii)  the answer is a whole-word span in exactly one
        supporting paragraph, and the other one does not
        contain it even after normalisation                   (D83)
  (iii) the question is of bridge type                        (D84)
"""

from __future__ import annotations

from dataclasses import dataclass

from inragwetrust.data.hotpotqa import HotpotExample, Paragraph
from inragwetrust.textmatch import contains_normalized, contains_span

FUNNEL_STAGES = ("all", "not_yes_no", "answer_in_exactly_one", "bridge_type")


@dataclass(frozen=True)
class PoolQuestion:
    """An eligible question with its passages already identified."""

    id: str
    question: str
    answer: str
    level: str
    answer_passage: Paragraph
    bridge_passage: Paragraph
    distractors: tuple[Paragraph, ...]


def is_yes_no(answer: str) -> bool:
    """Filter (i): True for the answers yes and no, in any capitalisation."""
    return answer.strip().lower() in {"yes", "no"}


def split_passages(ex: HotpotExample) -> tuple[Paragraph, Paragraph] | None:
    """Filter (ii): return (answer passage, bridge passage), or None if the rule fails."""
    sp = ex.supporting_paragraphs()
    if len(sp) != 2:
        return None
    hits = [contains_span(ex.answer, p.text) for p in sp]
    if sum(hits) != 1:
        return None
    answer_passage = sp[hits.index(True)]
    bridge_passage = sp[hits.index(False)]
    if contains_normalized(ex.answer, bridge_passage.text):
        return None
    return answer_passage, bridge_passage


def build_pool(examples: list[HotpotExample]) -> tuple[list[PoolQuestion], dict[str, int]]:
    """Apply filters (i)-(iii) in order; return the pool and the funnel counts."""
    funnel = dict.fromkeys(FUNNEL_STAGES, 0)
    pool: list[PoolQuestion] = []
    for ex in examples:
        funnel["all"] += 1
        if is_yes_no(ex.answer):
            continue
        funnel["not_yes_no"] += 1
        passages = split_passages(ex)
        if passages is None:
            continue
        funnel["answer_in_exactly_one"] += 1
        if ex.type != "bridge":
            continue
        funnel["bridge_type"] += 1
        answer_passage, bridge_passage = passages
        distractors = tuple(p for p in ex.paragraphs if p.title not in ex.supporting_titles)
        pool.append(
            PoolQuestion(
                id=ex.id,
                question=ex.question,
                answer=ex.answer,
                level=ex.level,
                answer_passage=answer_passage,
                bridge_passage=bridge_passage,
                distractors=distractors,
            )
        )
    return pool, funnel