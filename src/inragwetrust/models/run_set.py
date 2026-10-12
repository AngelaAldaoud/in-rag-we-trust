
"""Run all conditions of one model on one question split: greedy, then samples (D148-D149).

One model per process, so GPU memory is fully released when the process ends:
    python -u -m inragwetrust.models.run_set --model qwen --split pilot --out <Drive folder>
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # D102, set before vLLM is imported

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from inragwetrust.conditions import MODELS, condition_table
from inragwetrust.models import runs
from inragwetrust.models.chat import load_tokenizer
from inragwetrust.models.generator import MODES, Generator
from inragwetrust.retrieval.load import SPLITS, load_contexts

REPO = Path(__file__).resolve().parents[3]
CONDITIONS_PER_MODEL = 64  # 48 poisoned + 3 rho=0 + 1 closed-book + 12 BM25


def environment() -> dict:
    """Software and hardware versions stored in every record (D137), plus the exact code version."""
    import torch
    import transformers
    import vllm

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "vllm": vllm.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "commit": git("rev-parse", "HEAD"),
        "commit_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out", required=True, type=Path, help="run folder, e.g. .../runs/pilot")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    args = parser.parse_args(argv)

    env = environment()
    if "A100" not in env["gpu"]:
        raise SystemExit(f"D139: all pilot and main runs use the A100; this machine has {env['gpu']}")

    rows = load_contexts(args.split)
    conditions = [c for c in condition_table() if c.model == args.model]
    if len(conditions) != CONDITIONS_PER_MODEL:
        raise SystemExit(f"{args.model}: {len(conditions)} conditions, expected {CONDITIONS_PER_MODEL}")
    print(f"{args.model} | {args.split}: {len(rows)} questions x {len(conditions)} conditions | "
          f"commit {env['commit'][:7]}{' (dirty)' if env['commit_dirty'] else ''}", flush=True)

    tokenizer = load_tokenizer(args.model)
    start = time.time()
    generator = Generator(args.model)
    load_seconds = round(time.time() - start, 1)
    print(f"model loaded in {load_seconds} s", flush=True)

    for mode in args.modes:
        start = time.time()
        shards = runs.run_model(generator, tokenizer, rows, conditions, mode, args.out, env,
                                log=lambda line: print(line, flush=True))
        wall_seconds = round(time.time() - start, 1)
        done = sum(s["status"] == "done" for s in shards)
        print(f"{args.model} {mode}: {done} written, {len(shards) - done} skipped, {wall_seconds} s",
              flush=True)

        log_dir = args.out / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (log_dir / f"{args.model}_{mode}_{stamp}.json").write_text(json.dumps({
            "model": args.model, "split": args.split, "mode": mode, "questions": len(rows),
            "load_seconds": load_seconds, "wall_seconds": wall_seconds, "env": env,
            "shards": shards}, indent=2))


if __name__ == "__main__":
    main()