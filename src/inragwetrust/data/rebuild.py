"""Rebuild all CPU-reproducible data and verify it against the recorded checksums (D121).

Run with:
    uv run python -m inragwetrust.data.rebuild
GPU outputs (paraphrases, generated targets) are versioned files, not regenerated.
"""

from __future__ import annotations

import json
import sys

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.data.build import build as build_questions
from inragwetrust.data.freeze import freeze
from inragwetrust.data.hotpotqa import sha256_of
from inragwetrust.poisoning.build import build as build_poisons
from inragwetrust.retrieval.collection import OUT_DIR, build_collections


def _sha_in(meta_path, file_name):
    return json.loads(meta_path.read_text(encoding="utf-8"))["files"][file_name]["sha256"]


def main() -> int:
    results = []

    expected = _sha_in(PROCESSED_DIR / "questions_meta.json", "questions.jsonl.gz")
    build_questions()
    results.append(("B1 questions.jsonl.gz", expected, sha256_of(PROCESSED_DIR / "questions.jsonl.gz")))

    for name in ("paraphrases.jsonl.gz", "targets_generated.jsonl.gz"):
        sha = sha256_of(PROCESSED_DIR / name)
        results.append((f"GPU {name} (versioned)", sha, sha))

    expected = _sha_in(PROCESSED_DIR / "poisons_meta.json", "poisons.jsonl.gz")
    build_poisons()
    results.append(("B3 poisons.jsonl.gz", expected, sha256_of(PROCESSED_DIR / "poisons.jsonl.gz")))

    before = (PROCESSED_DIR / "sample.json").read_text(encoding="utf-8")
    freeze()
    after = (PROCESSED_DIR / "sample.json").read_text(encoding="utf-8")
    results.append(("B4 sample.json", "unchanged", "unchanged" if before == after else "CHANGED"))

    old = json.loads((OUT_DIR / "collections_meta.json").read_text(encoding="utf-8"))
    new = build_collections()
    for name, info in new["collections"].items():
        results.append((f"B4 collection_{name}", old["collections"][name]["sha256"], info["sha256"]))

    ok = True
    for label, exp, got in results:
        status = "OK" if exp == got else "MISMATCH"
        ok &= status == "OK"
        print(f"{status:8s} {label}")
    print("ALL DATA VERIFIED" if ok else "SOME OUTPUTS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())