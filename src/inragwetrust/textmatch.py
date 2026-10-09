"""Text matching rules shared by data filtering, poisoning and evaluation.

Every stage of the project answers "does this text contain this answer?"
with the functions below, so the rules can never differ between stages
(decisions D46, D83, D85).
"""

from __future__ import annotations

import re
import string

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    """Standard QA normalisation (D46).

    Lower-case, remove punctuation, remove the articles a/an/the, and
    collapse white space.
    """
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_normalized(answer: str, text: str) -> bool:
    """True if the normalised answer occurs as whole words in the normalised text."""
    a, t = normalize(answer), normalize(text)
    return bool(a) and f" {a} " in f" {t} "


def answer_pattern(answer: str) -> re.Pattern[str]:
    """Regular expression for the answer as a whole-word span, any capitalisation (D83)."""
    return re.compile(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", re.IGNORECASE)


def find_spans(answer: str, text: str) -> list[tuple[int, int]]:
    """Start and end positions of every whole-word occurrence of the answer."""
    if not answer.strip():
        return []
    return [m.span() for m in answer_pattern(answer).finditer(text)]


def contains_span(answer: str, text: str) -> bool:
    """True if the answer occurs in the text as a whole-word span (D83)."""
    return bool(find_spans(answer, text))