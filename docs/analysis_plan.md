# Analysis plan for the main run (frozen before any main result exists)

Frozen with the evaluator at git tag `evaluator-v1` (D180, D181). Applied unchanged to the 500 main questions.
Any deviation is logged as a revision with its reason. Sources: notes Sections 3.3–4.5 (Block A) and decisions D147–D179.

## 1. Data and responses
- 500 main questions (disjoint from the 50 pilot questions, D60); N stays 500 (D178).
- Primary metrics use greedy responses only (D50); the 5 samples at T = 0.7 serve self-consistency only (D171).
- Controlled contexts: rho in {0, 0.25, 0.5, 0.75, 1} x 4 poison types (rho = 0 shared, D41). Retrieval (BM25) and closed-book are analysed separately and never enter the confirmatory models.

## 2. Outcomes and metrics (code: `inragwetrust.evaluation`, tag `evaluator-v1`)
- Outcomes by the ordered rules D152–D156: Correct, Poisoned, Both, OtherWrong, Abstain, NoAnswer.
- Acc, ASR, Abst, NoAnswer, no-usable (Abst + NoAnswer); denominator = all responses (D153).
- Flag: Gemma judge, prompt `324e2f1c735e` (D168), over greedy responses with passages (D162, D173); unparseable labels excluded and counted.
- Hall: D166 rule; over all responses (primary) and over OtherWrong (D172); always with precision 9/13 and miss rate 0/18 (D169); Block A strict version as sensitivity.
- SC (D171), EM/F1 (D174, secondary).
- Acc+ / ASR+: exploratory upper bound, verdict prompt `87293ce31bb3` (D170, D175).

## 3. Intervals (Block A, D52)
- Single rates: Wilson 95%.
- Differences, SC, crossover points, kappa: paired bootstrap over questions, 10,000 resamples, project seed; all conditions of a question resampled together.

## 4. Confirmatory tests (Block A Section 4.4, D55): two-sided, alpha = 0.05, Holm within each family
GEE = logistic regression with generalised estimating equations, clustered by question, exchangeable working correlation, robust standard errors; rho entered as a number.

- **F1 (H1)**, 3 tests, one per model. Data: controlled greedy responses, all rho, all prompts and types. Model: `Poisoned ~ rho + prompt`. Test: rho coefficient != 0 (reported as odds ratio per unit rho with 95% CI).
- **F2 (H2)**, 18 tests (3 models x 2 meta-prompts x 3). Exact McNemar tests against the plain prompt:
  (a) Poisoned over all rho > 0 and poison types (pairs = question x type x rho, 8,000 per comparison);
  (b) at rho = 0, Correct;
  (c) at rho = 0, Abstain (deliberate abstention; no-usable reported alongside, exploratory).
  Because pairs in (a) share questions, a GEE estimate of the same contrast is reported as a sensitivity analysis.
- **F3 (H3)**, 3 tests, one per model. Data: controlled greedy responses with rho > 0 and a parseable flag. Model: `Poisoned ~ flag + rho + prompt + type`. Test: flag odds ratio != 1. Association only (D21). Caveats reported: the judge over-flags (dilutes the contrast) and agrees least for Qwen (kappa 0.57).
- **F4 (H4)**, 4 tests on the model pooling all three models. Data: controlled greedy responses with rho > 0. Model: `Poisoned ~ rho + prompt + type + model + rho:type + prompt:type + rho:model + prompt:model`. Tests: one Wald test per interaction block.

## 5. Exploratory analyses (intervals, no significance claims)
- Robustness curves per model x prompt x type; crossover point (smallest rho with ASR > Acc, D27) with bootstrap CI.
- Flag by rho with rho = 0 as baseline (pilot flags at rho = 0: Llama 25%, Qwen 27%, Mistral 2%); ASR among flagged vs unflagged per cell.
- Meta-prompt cost: Acc, Abst, no-usable at rho = 0; output tokens per answer.
- Retrieval conditions, read with the per-question effective rho of C4 (D144); closed-book separately.
- Hall (both denominators, strict sensitivity), SC, EM/F1, both-answers, format and truncation rates.
- The accuracy bracket: rule-based Acc/ASR vs Acc+/ASR+.
- Per-cell minimum detectable difference at 80% power from the observed p_d (D178).
- Position effects (D123, D127), B4 leak flags, GPU non-determinism (D139).

## 6. Fixed choices and known limitations
- No response is excluded; NoAnswer stays in every denominator.
- Levels of accuracy are uncertain within the bracket; comparisons use the same scoring everywhere.
- About a quarter of single cells have < 80% power for a 5-point difference (pilot p_d 75th pct 0.18).
- The GEE software version is pinned in Block F before the first fit and recorded.