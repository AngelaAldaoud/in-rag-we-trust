# Main run: launch instructions (Block E)

Code: git tag `evaluator-v1`. Order (D179): **greedy for all three models → judge → samples for all three models**, so every primary result exists before the long sampling phase.

## 0. Pre-flight
1. Mac: `git status -sb` clean; `git --no-pager log --oneline -1` shows the `evaluator-v1` commit.
2. Colab: A100 runtime. **CU balance ≥ 55** (D177); if lower, stop and reconsider the reserve pack.
3. Colab Secrets with notebook access: `GITLAB_TOKEN`, `HF_TOKEN`, `KAGGLE_API_TOKEN`.
4. Expected: about 6.5 h connected, 34–38 CU. Keep the tab open and the Mac awake (D78).

## 1. Notebook `06_e_main` (Colab)
- **Cell 1:** identical to notebook 04 cell 1; `git log` must show the `evaluator-v1` commit.
- **Cell 2:** identical to notebook 04 cell 2, with `RUN_DIR = Path("/content/drive/MyDrive/in-rag-we-trust/runs/main")`; it must print 0 shards (or the count already done, when resuming).
- **Cell 3 (greedy, all models, about 1.6 h):** the notebook 04 launch loop with the command
  `--model {model} --split main --out {RUN_DIR} --modes greedy`.
- **Cell 4 (check):** 192 greedy shards, 500 records each, commit = `evaluator-v1`, no `.tmp`.
- **Cell 5 (judge download):** notebook 05 cell 2 (Kaggle Gemma version 2).
- **Cell 6 (judge check):** `python -u -m inragwetrust.evaluation.judge check --run {RUN_DIR} --split main --gemma-dir {GEMMA_DIR}`; must end with **PASS**.
- **Cell 7 (judge run, about 25 min):** same with `run`; `None` count ≈ 0; meta shows prompts `324e2f1c735e` / `87293ce31bb3`.
- **Cell 8 (samples, all models, about 4.1 h):** the launch loop with `--modes samples`.
- **Cell 9 (check + archive):** 384 shards in total, 500 records each, commit = `evaluator-v1`; pack `runs/main` into `runs/main_runs.tar` (as notebook 04 cell 4). Note the CU balance, then **disconnect**.

## 2. Resume after a disconnection
Reconnect (A100), run cells 1–2, then re-run the interrupted cell: finished shards are skipped (D136). The judge has no partial resume: if cell 7 was interrupted, re-run it whole (about 25 min).

## 3. Mac
```bash
tar -xf ~/Downloads/main_runs.tar -C runs          # Drive may deliver it inside a zip: unzip first
uv run python -m inragwetrust.evaluation.score  --run runs/main --split main
uv run python -m inragwetrust.evaluation.report --run runs/main --split main --out results/main
```
Then Block F applies `docs/analysis_plan.md` unchanged.