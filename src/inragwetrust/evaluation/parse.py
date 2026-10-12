"""Final-answer extraction from model responses (D152) and the answer head (D156)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Answer:" or "Final Answer:" in any case, anywhere in a line, with optional Markdown stars
ANSWER_RE = re.compile(r"\**\b(?:final[ \t]+)?answer\b\**[ \t]*:\**[ \t]*(?P<rest>[^\n]*)",
                       re.IGNORECASE)
_EDGE = " \t*_`\"'\u201c\u201d\u2018\u2019"          # spaces, Markdown marks and quotes around the answer
_TRAILING = ".,;:!"
_LEADING_ANSWER = re.compile(r"^answer\b[\s:]*", re.IGNORECASE)   # "Answer: answer unknown" -> "unknown"
_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")               # one innermost (...) or [...]


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None       # None: no usable answer line (outcome NoAnswer, D153)
    position: str | None     # "line-start", "mid-line", or None when there is no answer line
    truncated: bool          # the response stopped at the output cap (finish_reason "length")
    cut: bool                # truncated and nothing follows the answer: it may be incomplete


def clean(raw: str) -> str:
    """Strip Markdown marks, quotes, trailing punctuation and a repeated 'Answer', until stable."""
    text, previous = raw, None
    while text != previous:
        previous = text
        text = _LEADING_ANSWER.sub("", text.strip(_EDGE).rstrip(_TRAILING))
    return text


def parse_answer(text: str, finish_reason: str) -> ParsedAnswer:
    """The text after the last 'Answer:' (D152); if that line is empty, the next non-empty line."""
    truncated = finish_reason == "length"
    match = None
    for match in ANSWER_RE.finditer(text):
        pass
    if match is None:
        return ParsedAnswer(None, None, truncated, False)

    line_start = text.rfind("\n", 0, match.start()) + 1
    before = text[line_start:match.start()].strip(" \t*#>-")
    position = "mid-line" if before else "line-start"

    rest = match.group("rest")
    if not clean(rest):
        rest = next((line for line in text[match.end():].split("\n") if clean(line)), "")
    answer = clean(rest)
    if not answer:
        return ParsedAnswer(None, position, truncated, False)
    cut = truncated and text.rstrip().endswith(rest.strip())
    return ParsedAnswer(answer, position, truncated, cut)


def head(answer: str) -> str:
    """The answer without bracketed remarks (D156): 'New York (assuming ...)' -> 'New York'."""
    text, previous = answer, None
    while text != previous:
        previous, text = text, _BRACKETED.sub(" ", text)
    if "(" in text:                                    # an unclosed remark runs to the end
        text = text[: text.index("(")]
    text = " ".join(text.split()).strip(" ,;:-")
    return text or answer