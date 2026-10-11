"""User-message texts for the prompt conditions and closed-book (C2, D128, D131-D132)."""

from __future__ import annotations

from inragwetrust.conditions import PROMPTS

TASK = "Answer the question using the passages below."
CLOSED_BOOK_TASK = "Answer the question."
EXTRA = {                         # the only difference between prompt conditions (D131)
    "plain": None,
    "consistency": "First check whether the passages agree with each other, then answer.",
    "reliability": "First judge how reliable each passage is, then answer.",
}
OUTPUT = (
    'Keep your reasoning brief. Finish with one final line in the form "Answer: <answer>", '
    "where <answer> is only a short name, date, number or phrase. If the question cannot "
    'be answered reliably, finish with "Answer: unknown".'
)


def format_passages(texts: list[str]) -> str:
    """Number the passages [1]..[n] in context order, separated by blank lines (D132)."""
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))


def user_message(question: str, passages: list[str], prompt: str) -> str:
    """The complete user message; an empty passage list gives the closed-book prompt."""
    if prompt not in PROMPTS:
        raise ValueError(f"unknown prompt: {prompt}")
    if not passages:
        if prompt != "plain":
            raise ValueError("closed-book uses the plain prompt only (A3)")
        return "\n\n".join([CLOSED_BOOK_TASK, f"Question: {question}", OUTPUT])

    task = TASK if EXTRA[prompt] is None else f"{TASK} {EXTRA[prompt]}"
    return "\n\n".join([task, "Passages:\n" + format_passages(passages),
                        f"Question: {question}", OUTPUT])


def messages(question: str, passages: list[str], prompt: str) -> list[dict]:
    """One user message and no system message (D128)."""
    return [{"role": "user", "content": user_message(question, passages, prompt)}]