# Dataset card — In RAG We Trust? (P5)

Controlled poisoned-retrieval data built from HotpotQA for measuring how open LLMs answer when part of the retrieved evidence is false.

## Source and licence
- HotpotQA (Yang et al., 2018), distractor setting, development set: 7,405 questions.
- Copy used: HuggingFace `hotpotqa/hotpot_qa`, commit `1908d6a`, SHA-256 `c20b638c…72f7c6` (the authors' server was unreachable on 9 Oct 2026).
- Licence: CC BY-SA 4.0 (inherited from HotpotQA).

## Construction
1. **Filters (B1):** not yes/no → 6,947; answer a whole-word span in exactly one supporting paragraph and absent from the other even after normalisation → 5,567; bridge questions only → 4,674; at least one safe target candidate → 4,672.
2. **Sampling:** seed 20261009; 50 pilot, 500 main, ordered reserve.
3. **Clean passages (B2):** the answer paragraph + 3 paraphrases by Gemma-2-9B-it (Kaggle version 2, bfloat16, vLLM 0.31.0), greedy, three rewriting instructions; automatic checks; two seeded retries; 77 questions replaced from the reserve.
4. **False answers (B4, D118):** written by Gemma-2-9B-it from the question, the true answer and the passage; must differ from and not contain the true answer and not occur in the question or passages. 589 of 602 generated; 13 fallback candidates.
5. **Poisons (B3):** every occurrence of the true answer replaced by the false answer; Swap adds nothing; Correct, Authority and Recency each add one sentence from five fixed templates.
6. **Validation (B4):** blocking checks (false answer present, true answer absent, exact framing); 3 main questions replaced.

## Final sample
- 50 pilot and 500 main questions (`data/processed/sample.json`).
- Per question: 4 clean passages, 16 poisoned passages (4 types × 4), 1 bridge passage, up to 8 distractors.
- BM25 collections: one per poison type, 9,301 documents each (2,200 clean, 2,200 poisoned, 550 bridge, 4,351 distractors).

## Files (`data/processed/`)
| File | Content |
|---|---|
| `questions.jsonl.gz` | 4,672 eligible questions: passages, entity type, target candidates, split |
| `paraphrases.jsonl.gz` | 700 questions × 3 paraphrases with check results (GPU output, versioned) |
| `targets_generated.jsonl.gz` | generated false answers with safety flag (GPU output, versioned) |
| `poisons.jsonl.gz` | 602 questions: clean and poisoned passages, false answer and its source |
| `sample.json` | the final 50 + 500 IDs, replacements, check report |
| `bm25/collection_<type>.jsonl.gz` | the four retrieval collections |
| `*_meta.json` | provenance: versions, settings, counts, checksums |

## Quality (manual checks by the author)
- Entity types: 25/30 correct (about 78% weighted).
- Paraphrases: 27/30 accepted ones faithful (errors change side facts, never the answer); 6/10 rejected ones were in fact faithful; 40/40 natural.
- False answers, same 30 questions before → after D118: plausible 5 → 19, grammatical 15 → 26, true answer visible 10 → 9.
- Automatic: no poison contains the true answer; 70 questions (12.7%) keep a proper-name word of the true answer elsewhere, 12 (2.2%) a number.

## Known limitations
- Bridge questions only; 14% of B2's sample were replacements, slightly favouring simpler passages.
- About 1 clean paraphrase in 10 changes a side fact; 6.4% omit a side name.
- About a third of false answers are implausible to a reader with world knowledge; in about 30% the passage context still points to the true answer.
- Template framing sentences are less natural than real misinformation.

## Reproduction
`uv sync` then `uv run python -m inragwetrust.data.rebuild` regenerates all CPU outputs and verifies their checksums. The GPU outputs are versioned; the notebooks `notebooks/01_b2_paraphrases.ipynb` and `notebooks/02_b4_targets.ipynb` show how they were produced.