"""Loading the HotpotQA development set (distractor setting).

The raw file is the Parquet copy of the dev split published in the
``hotpotqa/hotpot_qa`` HuggingFace repository, pinned to one commit and
verified by its SHA-256 checksum (decision D79, revised).
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

# ---------------------------------------------------------------- source
HF_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DEV_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    f"{HF_REVISION}/distractor/validation-00000-of-00001.parquet"
)
DEV_FILENAME = "hotpot_dev_distractor.parquet"
DEV_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
DEV_SIZE = 7405

# src/inragwetrust/data/hotpotqa.py -> parents[3] is the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# -------------------------------------------------------------- download
def sha256_of(path: Path) -> str:
    """Return the SHA-256 checksum of a file, reading it in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dev(raw_dir: Path = RAW_DIR) -> Path:
    """Download the dev file if it is missing, then verify its checksum."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / DEV_FILENAME
    if not path.exists():
        urllib.request.urlretrieve(DEV_URL, path)
    checksum = sha256_of(path)
    if checksum != DEV_SHA256:
        raise ValueError(
            f"Checksum mismatch for {path}: got {checksum}, "
            f"expected {DEV_SHA256}. Delete the file and download it again."
        )
    return path


# ---------------------------------------------------------- data classes
@dataclass(frozen=True)
class Paragraph:
    """One Wikipedia paragraph of a HotpotQA context."""

    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        """The whole paragraph.

        HotpotQA keeps the leading space of every sentence after the
        first, so the sentences are joined with an empty string.
        """
        return "".join(self.sentences)


@dataclass(frozen=True)
class HotpotExample:
    """One HotpotQA question with its ten context paragraphs."""

    id: str
    question: str
    answer: str
    type: str  # "bridge" or "comparison"
    level: str  # "easy", "medium" or "hard"
    supporting_titles: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]

    @classmethod
    def from_row(cls, row: dict) -> HotpotExample:
        """Build an example from one row of the Parquet table."""
        # A title can appear several times in supporting_facts (one entry
        # per sentence); keep each title once, in its first-seen order.
        supporting = tuple(dict.fromkeys(row["supporting_facts"]["title"]))
        ctx = row["context"]
        paragraphs = tuple(
            Paragraph(title=t, sentences=tuple(s))
            for t, s in zip(ctx["title"], ctx["sentences"])
        )
        return cls(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            type=row["type"],
            level=row["level"],
            supporting_titles=supporting,
            paragraphs=paragraphs,
        )

    def supporting_paragraphs(self) -> tuple[Paragraph, ...]:
        """The supporting paragraphs, in the order they appear in the context."""
        return tuple(p for p in self.paragraphs if p.title in self.supporting_titles)


# ------------------------------------------------------------------ load
def load_dev(raw_dir: Path = RAW_DIR) -> list[HotpotExample]:
    """Load and verify the full dev set as a list of HotpotExample."""
    path = download_dev(raw_dir)
    examples = [HotpotExample.from_row(r) for r in pq.read_table(path).to_pylist()]
    if len(examples) != DEV_SIZE:
        raise ValueError(f"Expected {DEV_SIZE} questions, found {len(examples)}.")
    return examples