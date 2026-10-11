"""Freeze the final pilot/main sample after the poison checks (decision D117).

Run with:
    uv run python -m inragwetrust.data.freeze
"""

from __future__ import annotations

import json

from inragwetrust.data.build import PROCESSED_DIR
from inragwetrust.poisoning.build import read_jsonl_gz
from inragwetrust.poisoning.checks import report


def freeze(out_dir=PROCESSED_DIR) -> dict:
    """Replace questions with blocking poison failures; write sample.json."""
    path = out_dir / "poisons.jsonl.gz"
    blocked = report(path)["questions_with_blocking_failures"]
    records = sorted(read_jsonl_gz(path), key=lambda r: r["rank"])
    reserve = iter([r["id"] for r in records
                    if r["final_split"] == "reserve" and r["id"] not in blocked])

    sample, replaced = {"pilot": [], "main": []}, []
    for r in records:
        split = r["final_split"]
        if split not in sample:
            continue
        if r["id"] in blocked:
            substitute = next(reserve)
            sample[split].append(substitute)
            replaced.append({"replaced": r["id"], "by": substitute, "split": split,
                             "reason": blocked[r["id"]]})
        else:
            sample[split].append(r["id"])

    final_ids = set(sample["pilot"]) | set(sample["main"])
    out = {
        "pilot": sample["pilot"],
        "main": sample["main"],
        "b4_replacements": replaced,
        "poison_checks_final_sample": report(path, ids=final_ids),
    }
    (out_dir / "sample.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    return out


if __name__ == "__main__":
    out = freeze()
    print("pilot:", len(out["pilot"]), "| main:", len(out["main"]))
    print("B4 replacements:", out["b4_replacements"])
    checks = out["poison_checks_final_sample"]
    print("final sample:", json.dumps({k: checks[k] for k in
          ("blocking_failures", "questions_with_partial_leak", "swap_length_ratio")}))