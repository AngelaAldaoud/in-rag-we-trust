"""Automatic checks for clean paraphrases (decision D108).

A paraphrase passes when the returned list of failed checks is empty.
"""

from __future__ import annotations

import re

from inragwetrust.textmatch import contains_span, normalize

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NAME = re.compile(r"(?<![\w'’])[A-Z][\w'’\-]*")
US_FORMS = {"us", "usa", "america", "american", "americans", "united", "states"}
SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][\w'’\-]*)")
WORD = re.compile(r"[\w'’\-]+")
CHATTER_STARTS = ("here is", "here's", "here are", "sure", "rewritten", "paraphrase")
MIN_RATIO, MAX_RATIO = 0.7, 1.4
PREFIX = 5
NON_BLOCKING = {"missing_name", "missing_number"}   # recorded, not rejecting (D111)


def _prep(text: str) -> str:
    """Write U.S. / U.S.A. as US before tokenising (D110)."""
    return re.sub(r"\bU\.\s?S\.(?:\s?A\.)?", "US", text)


def _key(token: str) -> str:
    """Lower case, no possessive 's, no d'/l' elision, one form for the United States."""
    t = re.sub(r"['’]s$", "", token.lower()).strip("'’-")
    t = re.sub(r"^[dl]['’]", "", t)
    return "united-states" if t in US_FORMS else t


def numbers(text: str) -> set[str]:
    """All numbers in the text, with thousands separators removed."""
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def names(text: str) -> set[str]:
    """Capitalised words (at least two letters) that are not at the start of a sentence."""
    text = _prep(text)
    starts = {m.start(1) for m in SENTENCE_START.finditer(text)}
    keys = {_key(m.group()) for m in NAME.finditer(text) if m.start() not in starts}
    return {k for k in keys if len(k) >= 2}


def words(text: str) -> set[str]:
    return {_key(w) for w in WORD.findall(_prep(text))}


def _known(token: str, vocabulary: set[str]) -> bool:
    """True if the vocabulary has the token, or a word with the same first PREFIX letters."""
    if token in vocabulary:
        return True
    return len(token) >= PREFIX and any(v[:PREFIX] == token[:PREFIX] for v in vocabulary)


def title_words(title: str, original: str) -> list[str]:
    """Capitalised words of the title (no brackets) that occur in the original passage."""
    core = re.sub(r"\s*\(.*?\)", "", title)
    vocabulary = words(original)
    keys = [_key(t) for t in WORD.findall(_prep(core)) if t[0].isupper()]
    return [k for k in keys if len(k) >= 2 and _known(k, vocabulary)]


def accepted(failed: list[str]) -> bool:
    """A paraphrase is accepted when none of its failures is blocking (D111)."""
    return all(f in NON_BLOCKING for f in failed)


def check_paraphrase(original: str, paraphrase: str, gold_answer: str,
                     finish_reason: str = "stop", title: str | None = None) -> list[str]:
    """Return the names of the failed checks (empty list = accepted)."""
    failed = []
    if finish_reason != "stop":
        failed.append("truncated")
    if paraphrase.lower().startswith(CHATTER_STARTS) or "**" in paraphrase:
        failed.append("chatter")
    if not contains_span(gold_answer, paraphrase):
        failed.append("answer_missing")
    if normalize(paraphrase) == normalize(original):
        failed.append("identical")
    ratio = len(paraphrase) / max(len(original), 1)
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        failed.append("length")
    n_orig, n_para = numbers(original), numbers(paraphrase)
    if n_para - n_orig:
        failed.append("new_number")
    if n_orig - n_para:
        failed.append("missing_number")
    w_orig, w_para = words(original), words(paraphrase)
    if any(not _known(n, w_orig) for n in names(paraphrase)):
        failed.append("new_name")
    if any(not _known(n, w_para) for n in names(original)):
        failed.append("missing_name")
    if title is not None and any(not _known(t, w_para) for t in title_words(title, original)):
        failed.append("title_missing")
    return failed