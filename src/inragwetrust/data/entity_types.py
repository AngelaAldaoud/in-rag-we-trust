"""Entity type of each gold answer, from spaCy NER in context (D87, D89, D91).

The answer passage is processed by spaCy; the entity whose character span
overlaps the answer the most gives the fine label (e.g. PERSON, GPE), which
is mapped to one of six coarse types used to choose target answers.
"""

from __future__ import annotations

from functools import lru_cache

import spacy

from inragwetrust.textmatch import find_spans

SPACY_MODEL = "en_core_web_sm"
NO_ENTITY = "NONE"

COARSE_TYPE = {
    "PERSON": "person",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "DATE": "date",
    "TIME": "date",
    "CARDINAL": "number",
    "QUANTITY": "number",
    "MONEY": "number",
    "PERCENT": "number",
    "ORDINAL": "number",
    "ORG": "organisation",
}


@lru_cache(maxsize=1)
def get_nlp() -> spacy.language.Language:
    """Load the spaCy model once; parser and lemmatiser are not needed for NER."""
    return spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer"])


def label_in_doc(answer: str, doc: spacy.tokens.Doc) -> str:
    """Fine label of the entity that best covers the first occurrence of the answer.

    The entity must overlap at least half of the answer's characters;
    otherwise the answer has no entity label (NO_ENTITY).
    """
    spans = find_spans(answer, doc.text)
    if not spans:
        return NO_ENTITY
    start, end = spans[0]
    best_label, best_overlap = NO_ENTITY, 0
    for ent in doc.ents:
        overlap = min(end, ent.end_char) - max(start, ent.start_char)
        if overlap > best_overlap:
            best_label, best_overlap = ent.label_, overlap
    if 2 * best_overlap < end - start:
        return NO_ENTITY
    return best_label


def coarse_type(fine: str) -> str:
    """Map a spaCy label to person, place, date, number, organisation or other."""
    return COARSE_TYPE.get(fine, "other")


def tag_answers(items: list[tuple[str, str]], batch_size: int = 64) -> list[str]:
    """Fine labels for (answer, passage) pairs, processed in batches.

    Answers with no entity in their passage are tagged once more on their
    own text, with the same half-coverage rule (D93).
    """
    nlp = get_nlp()
    docs = nlp.pipe((passage for _, passage in items), batch_size=batch_size)
    labels = [label_in_doc(answer, doc) for (answer, _), doc in zip(items, docs)]
    missing = [i for i, label in enumerate(labels) if label == NO_ENTITY]
    alone = nlp.pipe((items[i][0] for i in missing), batch_size=batch_size)
    for i, doc in zip(missing, alone):
        labels[i] = label_in_doc(items[i][0], doc)
    return labels