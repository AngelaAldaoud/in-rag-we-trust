# Presentation notes — how to explain each result to Prof. Ferrara

One entry per result, table or plot (Rule 10). Each entry has four parts.

## Template

### <Figure/Table name>
- **What it shows:**
- **How to read it:**
- **What it means:**
- **Its limitation:**

---

<!-- Entries are added from Block D onward, one per result. -->
### Finding: early environment check (B2)
- **What it shows:** before generating any data, I installed the GPU stack on Colab and tested one prompt. The check revealed that the current vLLM (0.31.0) no longer supports the 4-bit quantisation I had planned.
- **How to read it:** the method list printed by vLLM itself has no "bitsandbytes" entry, so it's a verified fact, not a guess.
- **What it means:** I switched all models to native 16-bit precision. That's fairer (one precision for everyone) and removes quantisation as a confounder. Because the check came first, no experiment had to be redone.
- **Its limitation:** 16-bit needs more GPU memory, so the full runs depend on the A100; on smaller GPUs only the L4 fallback works.

### Result: model-based choice of false answers — a negative result (B2)
- **What it shows:** for 6 test questions, the random false answer next to the one chosen by the helper model, in two prompt versions.
- **How to read it:** only 1 of 6 choices was clearly better (Bill Paxton for Daniel Radcliffe), and one choice changed from good to bad between versions.
- **What it means:** the model couldn't reliably improve plausibility, mainly because when the answer's type is wrong, no good candidate exists. I kept the reproducible random draw. Since every question keeps the same false answer in all conditions, this affects the *level* of attack success, not the *comparisons* my research questions rely on.
- **Its limitation:** absolute attack success may be underestimated; the share of plausible false answers is measured by hand in B4.

### Finding: a strict data filter creates selection bias (B2)
- **What it shows:** how many of the 500 main questions keep all three paraphrases under three versions of the automatic checks: 341 (strict), 357 (tokenisation fixed), 430 (omissions allowed).
- **How to read it:** each failed question is replaced from the reserve. Under the strict rules a third of the sample would have been replaced, and always the passages richest in names and dates.
- **What it means:** a quality filter that looks harmless can silently make the sample easier. I changed the rule to what the definition of a clean passage actually requires: keep the answer and the main entity, add nothing. Small omissions are allowed, recorded and checked by hand.
- **Its limitation:** both changes came after seeing the data; 14% of questions were still replaced, slightly favouring simpler passages; and accepted paraphrases can be less detailed than the original.


### Result: quality of the clean paraphrases (B2)
- **What it shows:** 40 paraphrases I judged blind: 30 that the checks accepted, 10 they rejected.
- **How to read it:** 27 of 30 accepted ones are faithful; the 3 errors change a side fact, never the answer. 6 of 10 rejected ones were actually fine.
- **What it means:** clean passages always support the true answer; about 1 in 10 carries a small side error. The automatic checks are strict: they waste some good paraphrases but let few bad ones through.
- **Its limitation:** 30 examples give a wide range (about 3–26%) for the error rate.

### Fact: cost of generating the data (B2)
- 2,100 paraphrases in 26 minutes on an L4 GPU, at about 1.5 compute units per hour; generation, retries and checks cost about 3.2 units.
- At 16-bit precision the helper model leaves little memory for parallel work, so generation was slower than planned; the A100 for the main runs has much more memory.

### Example: one passage, four poisons (B3)
- **What it shows:** the clean sentence "…starring David Haig and Daniel Radcliffe…" and its four poisoned versions.
- **How to read it:** Swap changes only the name; Correct, Authority and Recency add one sentence each ("other sources are wrong", "official sources confirm", "latest update").
- **What it means:** because the types differ by one sentence, any difference in how often models believe them is caused by that sentence: the core of RQ4.
- **Its limitation:** template sentences are less natural than real misinformation; some false answers are less plausible than a human would write.

### Finding: false dates must look like dates (B3)
- **What it shows:** "released to theatres on April 20, 1946" became "…on 1902" under the first rule, and "…on September 25, 2017" after the fix.
- **How to read it:** for dates and numbers, the false answer now has the same form as the true one; 43 of 550 targets changed.
- **What it means:** an ungrammatical lie is easy to reject and would make models look more robust than they are.
- **Its limitation:** other mentions of the truth can remain ("a 1946 film"); B4 measures how often.


### Finding: believable lies need context, not just the right type (B4)
- **What it shows:** 30 poisons judged by hand twice: with false answers of the same entity type (5/30 plausible) and with false answers written to fit the question and passage (19/30 plausible). Grammatical: 15/30 → 26/30.
- **How to read it:** same 30 questions, paired; 15 improved and 1 got worse (McNemar p ≈ 0.0005).
- **What it means:** with mostly absurd lies, every model would look robust — a floor effect hiding all comparisons. Context-fitting false answers make the attack realistic; every one still passes the same safety rules, so none can contain the truth.
- **Its limitation:** about a third remain implausible to someone with world knowledge, and in about 30% the passage's context still points to the true answer (e.g. "007" → James Bond).

### Claim: the data is reproducible byte for byte (B4)
- **What it shows:** one command rebuilds every CPU-generated file of the dataset and prints OK for each checksum; "ALL DATA VERIFIED".
- **How to read it:** each file is compared with the fingerprint recorded when it was first built.
- **What it means:** anyone can reconstruct exactly my 550 questions, their passages, poisons and retrieval collections.
- **Its limitation:** the two GPU outputs (paraphrases, false answers) are stored files, not regenerated, because sampled GPU generation isn't guaranteed identical across software versions.

### Design: how a context is built (C1)
- **What it shows:** each question's context is 4 answer passages + 1 bridge passage; at ratio ρ, 4ρ of the answer passages are swapped for their poisoned twins.
- **How to read it:** one fixed random order per question decides which passages get poisoned first and where every passage sits; raising ρ only switches more passages from clean to poisoned, nothing moves.
- **What it means:** any difference between ρ values, poison types, prompts or models comes from that factor alone — the comparisons are paired.
- **Its limitation:** position effects can only be compared across questions, since the order is fixed within a question.

### Table: the 192 conditions (C1)
- **What it shows:** 144 poisoned (3 models × 3 prompts × 4 types × 4 ratios), 9 clean baselines, 3 closed-book, 36 realistic-retrieval conditions.
- **How to read it:** the clean baseline is shared by all poison types because at ρ = 0 there is no poison; closed-book uses only the plain prompt.
- **What it means:** one table, generated by code, drives generation, scoring and analysis, so no stage can use a different design.
- **Its limitation:** retrieval conditions have no controlled ρ — BM25 decides what is retrieved.

### Check: are poisons spread evenly over positions? (C1)
- **What it shows:** over 500 questions, the first poison lands at positions 0–4 107, 95, 82, 99, 117 times; the bridge 113, 107, 92, 96, 92 times.
- **How to read it:** about 100 each is expected; chi-square tests find no imbalance (p ≈ 0.14 and 0.46).
- **What it means:** no condition is helped or hurt by "Lost in the Middle" position effects on average.
- **Its limitation:** the original passage is poisoned first in 102 questions instead of about 125 (p ≈ 0.02); a chance result I reported rather than re-rolling the seed.

### Design: the three prompts (C2)
- **What it shows:** plain, consistency and reliability prompts; the last two add one sentence each, everything else is identical.
- **How to read it:** the passages, question and "Answer:" format are the same text in all three, and the whole prompt is one user message for every model.
- **What it means:** any difference in attack success between prompts comes from that one sentence (RQ2).
- **Its limitation:** each model keeps its own wrapper — Qwen's default system line, Llama's fixed date header — constant within a model, but different between models.

### Table: prompt lengths (C2)
- **What it shows:** mean 673–746 tokens, maximum 1,783, across 28,050 prompts per model.
- **How to read it:** same text, different counts, because each model splits words into its own pieces (Mistral's vocabulary is smallest).
- **What it means:** all prompts fit easily; reading the prompts dominates the compute cost, which is why the 5 samples share one prompt reading.
- **Its limitation:** the BM25 prompts (C4) are measured separately and may be longer.

### Table: smoke test across the three models (C3)
- **What it shows:** 3 pilot questions × 52 conditions per model: how often answers end with a proper "Answer:" line, how often they say "unknown", how often they are cut off, and how long they are.
- **How to read it:** Qwen abstains more as poison grows (2/9 at ρ=0, 18/36 at ρ=1); Llama writes long reasoning; Mistral is brief but writes "Answer:" inside sentences.
- **What it means:** the pipeline works end to end for all models, and the smoke test exposed issues *before* the main run — a cut-off problem (fixed) and two parsing cases for the evaluator.
- **Its limitation:** only 3 questions; rates are illustrations, not estimates — the pilot (50 questions) gives the real figures.

### Finding: the output cap was cutting off the model's reasoning (C3)
- **What it shows:** with a 256-token cap, 17% of Llama's answers were cut off before the final answer — 35% under the consistency prompt. At 512 tokens: 5%.
- **How to read it:** same 3 questions, same prompts, same seeds; only the cap changed.
- **What it means:** the cap is part of the measuring instrument. If it censors answers mainly under one prompt, the prompt comparison measures the cap. I fixed the instrument and left the prompts (the treatment) untouched; making prompts stricter or putting the answer first would have changed what I study.
- **Its limitation:** the remaining 5% are repetition loops and endless comparisons under conflict, which no cap fixes; they are reported as answers without a final answer.


### Table: throughput and compute plan (C3)
- **What it shows:** requests per second per model on the A100 (Qwen 23, Llama 12.6, Mistral 21 greedy; 4–10 with 5 samples) and Qwen on the L4 (4.5× slower). The whole study needs about 6 A100 hours of generation, about 46 of my 80 compute units including the judge.
- **How to read it:** samples are cheaper than 5 separate answers because the 5 share one reading of the prompt; Llama is slowest because it writes the longest answers.
- **What it means:** the full design fits the budget with about 40% margin, without cutting anything; the L4 would take 27 hours, so the A100 is required.
- **Its limitation:** retrieval prompts and the judge are not yet measured; the pilot re-measures everything.

### Observation: the GPU itself can change an answer (C3)
- **What it shows:** with identical prompts and seeds, a few of Qwen's greedy answers differed between the A100 and the L4.
- **How to read it:** GPUs do floating-point arithmetic in different orders; tiny differences can tip which token is "most likely".
- **What it means:** all main runs use one GPU type, so this cannot add noise between conditions.
- **Its limitation:** exact output reproducibility is therefore not claimed; settings, seeds and versions are recorded instead.

### Table: what a real retriever returns (C4)
- **What it shows:** for 500 questions, BM25 searched a 9,301-passage collection containing true, poisoned and unrelated passages, and returned the top 5.
- **How to read it:** half the questions retrieve no passage containing the answer (BM25 finds the first hop, which names the entity in the question, not the second); when answer passages are found, poisons come with them at an average ratio of 0.29–0.50.
- **What it means:** realistic retrieval produces exactly the mixed low-to-middle ratios my controlled design studies; ρ = 0.75–1 is a stress test. And because retrieval often misses the answer, the controlled contexts always include it — the study measures how models use evidence that is there.
- **Its limitation:** a small collection with planted near-duplicates — a poisoned-corpus scenario (PoisonedRAG), not natural web exposure.

### Finding: how a tie is broken can create a result (C4)
- **What it shows:** 62% of Swap poisons score exactly like their true twin; my first rule broke ties alphabetically, which always favoured the truth for Swap. With a fair, seeded tie-break, Swap's effective poison ratio went from 0.42 to 0.50, and questions with a poison ranked first from 25 to 58.
- **How to read it:** same searches, same scores; only the order among exact ties changed. Framed poisons barely changed, because their extra sentence prevents ties.
- **What it means:** BM25 cannot tell an entity swap from the truth — and an arbitrary implementation detail had hidden this. I found it by asking which choices could favour one condition.
- **Its limitation:** the random tie-break is one fair rule; a different seed would move a few individual questions, not the averages.

### Observation: persuasive framing trades retrievability (C4)
- **What it shows:** Swap poisons are retrieved like the truth (ratio 0.50); Correct, Authority and Recency less (0.29–0.36).
- **How to read it:** the added sentence makes the passage longer and adds no question words, so BM25 scores it lower.
- **What it means:** framing that may persuade a model more also reaches the model less often — relevant when reading RQ4.
- **Its limitation:** specific to BM25; a dense retriever might behave differently.


### Result: the pilot run (D1)
- **What it shows:** the full design (64 conditions × 3 models, greedy + 5 samples) ran once on 50 held-out pilot questions: 384 complete shards, 57,600 answers, 47 minutes on an A100, 6.5 compute units.
- **How to read it:** one row per model; load = putting the model on the GPU, greedy/samples = generation time; out-tokens = total words-pieces the model wrote.
- **What it means:** the pipeline works end to end, unattended, and every answer is traceable to one commit; Llama is the slow model because its occasional repetition loops make each batch wait.
- **Its limitation:** 50 questions only — the pilot is for building and checking the evaluator and measuring cost, not for conclusions; pilot timing does not scale linearly to the main run.

### Result: meta-prompts make answers longer (D1, quantified in D4)
- **What it shows:** asking the model to check consistency or source reliability makes it write about 2–4 times more tokens than the plain prompt, for all three models.
- **How to read it:** tokens per 50 answers, by prompt; higher = more reasoning written before the answer line.
- **What it means:** any robustness gain from the meta-prompts (RQ2) has a price in time and compute; we report both.
- **Its limitation:** length is a cost measure, not a quality measure — a longer answer is not necessarily a better-reasoned one.

### Finding: I audited my own evaluation by hand — and it changed the method (D2)
- **What it shows:** a blind hand check of 60 scored answers and 20 "hallucinations". When my rules said Correct, Poisoned or Abstain, they were right 35/35. But only 10 of 20 "other wrong" answers were really wrong — 7 were correct and 3 poisoned, written differently (IWC vs International Watch Co., Merck vs Merck & Co.). And only 4 of 20 flagged hallucinations were real; 0 of 10 for Mistral.
- **How to read it:** each row is a rule outcome; the fraction is how often my own reading agreed; the interval is the 95% Wilson confidence interval.
- **What it means:** string matching alone would have underestimated accuracy by about 7–10 points and attack success by 3–5, and would have called Mistral the worst hallucinator just because it writes long answers. So I kept the rules where they are reliable, added a validated judge only for the "other wrong" cases, and redefined hallucination as invented facts (names, numbers, dates) found with spaCy — all decided on the pilot, before any main result.
- **Its limitation:** small samples (60 and 20), so the error rates are rough (OtherWrong agreement 30–70%); the hallucination check was not blind; a wrong answer with no name or number is never counted as a hallucination.
- **If asked "why not just use the judge for everything?":** the rules were perfect on the positive outcomes and are transparent and free; I used the model only where the audit proved the rules fail.


### Method: how a free-text answer becomes an outcome (D2)
- **What it shows:** each answer is read by a fixed rule list — no answer line → NoAnswer; "unknown" → Abstain; contains both truth and lie → Both; truth → Correct; lie → Poisoned; refusal phrase → Abstain; otherwise OtherWrong (re-checked by the judge).
- **How to read it:** rules apply top to bottom; the first that fits decides. Bracketed side remarks are ignored, so the model is scored on what it committed to.
- **What it means:** rules were written after reading all 57,600 pilot answers — e.g. 22% of Mistral's answers put "Answer:" mid-line, and Llama skips the answer line 9% of the time, so the parser handles both and NoAnswer is reported separately (a format failure is not caution).
- **Its limitation:** text rules cannot see aliases ("IWC" = "International Watch Co."); the audit measured this, which is why OtherWrong is re-checked by the judge.


### Result: can a model judge be trusted? Validated against my blind labels (D3)
- **What it shows:** Gemma flags "the passages disagree" in agreement with my 300 blind labels at kappa 0.735 (95% CI 0.649–0.814), above the 0.6 bar; raw agreement 89%.
- **How to read it:** kappa removes agreement that would happen by chance; 0 = chance, 1 = perfect; 0.6+ = substantial.
- **What it means:** the conflict flag for RQ3 is a validated measurement, not an assumption. It errs in one direction — it says "yes" on 14% of responses I judged "no" — so flag rates are a bit high and flagged/unflagged differences are, if anything, understated.
- **Its limitation:** weaker for Qwen (kappa 0.57 on 100 items); one judge model, one annotator.

### Finding: my first judge prompt was biased — and the second one I kept out of the main results
- **What it shows:** asked "gold, other or neither?", the judge almost never said "other", even when the answer was exactly the target. With neutral labels A/B, it passed 20/20 known answers and agreed with my labels at kappa 0.84 — but it still credited the gold too often (92.8% < 95% on poisoned answers).
- **How to read it:** the acceptance rule was fixed before the result; one check failed, so the judge does not change my main outcomes.
- **What it means:** main results use transparent rules (which slightly undercount correct answers); the judge gives an upper bound; the truth lies between.
- **Its limitation:** the bracket can be wide where models write long, alias-heavy answers.

### Finding: models given passages rarely invent facts
- **What it shows:** with a validated rule (precision 9/13, no misses in 18 checks), only 0.4–2.5% of wrong answers contain a name, number or date absent from the passages.
- **How to read it:** a "hallucination" here means a fact the model could not have read in its context.
- **What it means:** wrong answers mostly reuse context facts in the wrong way rather than inventing — which is exactly what poisoning exploits.
- **Its limitation:** small validation samples (13 and 18); wrong answers without names or numbers are not counted.

### Result: the pilot robustness curves (D4)
- **What it shows:** for each model, accuracy and attack success as the share of poisoned passages (rho) goes from 0 to 1.
- **How to read it:** at rho = 0 there is no poison, so attack success must be 0 (it is); each step right adds one poisoned passage out of four.
- **What it means:** for Qwen and Mistral, one poisoned passage barely matters — the drop comes when the lie becomes the majority; Llama is fooled already by one (6.5%).
- **Its limitation:** 50 questions (±4–8 points); accuracy levels sit inside a bracket (rules 54–59% vs judge 75–77% at rho = 0), comparisons are more reliable than levels.

### Result: when do models notice a conflict?
- **What it shows:** the share of answers saying the passages disagree peaks at mixed contexts and falls when every passage tells the same lie.
- **How to read it:** at rho = 1 there is nothing to disagree about — all passages agree on the false answer.
- **What it means:** conflict detection works only when truth and lie coexist; a uniform lie is invisible. Even clean contexts get flagged (Llama 25%, Qwen 27%), so rho = 0 is the baseline.
- **Its limitation:** the flag judge over-flags slightly (validated, kappa 0.735).

### Result: protective prompts cost answers under real retrieval
- **What it shows:** with BM25 retrieval, attack success stays low, but asking models to check consistency or reliability makes them refuse far more (Mistral about 24% → 62% no usable answer).
- **How to read it:** "no usable answer" = abstained or no answer line.
- **What it means:** RQ2's price — caution against poison also blocks answers when the evidence is merely incomplete.
- **Its limitation:** pilot only; BM25 often retrieves no answer-bearing passage, which itself drives refusals.



99 & Power check
  & Script measuring $p_d$ on the pilot and McNemar power; analysis of the options; decision D178 proposed
  & Run by the student; decision accepted by the student \\
100 & Analysis plan and launch instructions
  & \texttt{docs/analysis\_plan.md}, \texttt{docs/main\_run.md}; D179--D181
  & Checked against Block A Sections 3.3--4.5 and decisions D147--D178; accepted by the student \\
101 & Block D close
  & State reports, notes, BLOCK\_D handoff
  & Reviewed by the student \\



  ### Finding: I checked whether my sample was big enough — and documented where it is not (D5)
- **What it shows:** using the pilot, I measured how often two prompts disagree on the same question (p_d). The median cell has 92% power to detect a 5-point difference at N = 500; a quarter of cells only 75%.
- **How to read it:** "power" = the chance of detecting a real 5-point difference; 80% is the usual target. MDD = the smallest difference detectable with 80% power: about 4 points in a typical cell, at most about 6.6 points in the weakest.
- **What it means:** my pre-registered rule said "increase N if under-powered". Only 52 more prepared questions existed, which would not reach 80% anyway, so I kept N = 500 as a documented deviation. The confirmatory tests pool 16 cells per question, so they are well powered; single cells are exploratory.
- **Its limitation:** in single cells I can miss effects smaller than about 5–7 points, so a non-significant single-cell result does not mean "no effect".
- **If asked "why didn't you just add questions?":** the only ready questions were 52 reserve ones (power 0.79, still below 80%), and adding them meant rebuilding the data and retrieval pipeline the night before the main run — a bigger risk to correctness than the small power gain.

## Block E · E1 Preflight (how to explain it to Prof. Ferrara)
- **What it shows:** before the main run, I verified and recorded the code version, budget and storage.
- **How to read it:** the notebook prints the commit (a93ec0c = tag evaluator-v1) and 0 changed files; compute 65.5 units, above my 55-unit threshold; storage far above the need.
- **What it means:** every main result comes from the frozen, validated code; the run was not at risk of running out of budget or space.
- **Limitation:** the compute figures are Colab's own counter, read by hand at phase boundaries.
## Block E · E2 Greedy run (how to explain it to Prof. Ferrara)
- **What it shows:** the 96,000 primary answers (3 models × 64 conditions × 500 questions) were generated in 1 h 54 min for 10.2 compute units, all by the frozen code.
- **How to read it:** the table gives load time, generation time and output tokens per model; Llama is slowest because it writes the most.
- **What it means:** every shard is complete and every condition contains the same 500 questions, so all my comparisons are paired, question by question.
- **Limitation:** greedy decoding on a GPU is deterministic only up to small numerical differences (D139); a rerun on other hardware might differ in rare answers.


## Block E · E3 Judge (how to explain it to Prof. Ferrara)
- **What it shows:** a separate model (Gemma-2-9B-it) labelled every main greedy answer: 94,500 conflict flags and 93,547 verdicts, with the exact prompts I validated against my own hand labels.
- **How to read it:** before the full run, a check on 20 answers with a certain correct label had to score at least 19/20; it scored 19/20, and the one miss is the error type I had already measured.
- **What it means:** the flag, the measurement behind RQ3, comes from a validated instrument (kappa 0.735) applied unchanged to the main data.
- **Limitation:** the judge over-flags and agrees least for Qwen; the verdict under-recognises the false target, which is why it only gives an upper bound.

## Block E · E4 Samples and archive (how to explain it to Prof. Ferrara)
- **What it shows:** 480,000 sampled answers (5 per prompt, T = 0.7) for self-consistency, generated in 4 h 14 min; the whole main run used 39.1 compute units.
- **How to read it:** each model's samples come from one uninterrupted process; every shard was checked, and the complete output was archived with a SHA-256 fingerprint.
- **What it means:** the data I score on my laptop is provably the data the GPU produced (the fingerprint must match), and nothing is missing.
- **Limitation:** sampling uses one fixed seed per process; a different seed would give different samples, which affects individual answers but not the self-consistency measure in expectation.


## Block E · E5 Main result tables (how to explain them to Prof. Ferrara)
*"What it means" is added in Block F, under the frozen analysis plan.*

**Integrity (before any table).** The scored data is byte-identical to the GPU output (equal SHA-256), and it was scored with the frozen code (no code change since the tag; 293 tests pass).

- **grid.csv:** *Shows* every greedy metric per model × prompt × poison type × ρ (153 cells of 500 answers). *How to read:* one row per cell; each rate has a 95% Wilson interval (`_lo`, `_hi`) and its denominator (`_n`). *Limitation:* single cells are exploratory; about a quarter are under-powered for a 5-point difference (D178).
- **by_rho.csv / by_rho_pooled.csv:** *Shows* the same metrics pooled over poison types (and, in the pooled file, over prompts) for each ρ. *How to read:* the robustness curve; read across ρ within a model. *Limitation:* pooling hides type differences; the confirmatory tests are the GEE and McNemar models, not these tables.
- **retrieval.csv:** *Shows* answers when a real BM25 search chose the passages. *How to read:* compare with the controlled conditions at the effective ρ BM25 produced (D144). *Limitation:* the share of poison varies per question, so these cells are not a fixed ρ.
- **closed_book.csv:** *Shows* answers with no passages at all. *How to read:* what the models know without retrieval. *Limitation:* without passages, hallucination counts every fact as invented (D172).
- **sc.csv:** *Shows* self-consistency over the 5 samples (by normalised answer and by outcome, D171). *How to read:* 1.0 means the same answer in all 5 samples. *Limitation:* it depends on T = 0.7; it measures stability, not correctness.
- **format.csv:** *Shows* where the answer line sits, how often it is missing, and how often the 512-token cap was hit. *How to read:* Llama hits the cap in 5.1% of greedy answers (repetition loops), which explains most of its NoAnswer. *Limitation:* format failures are counted, never dropped, so they lower every rate for that model.
- **Sanity finding (D188):** two of 500 questions are flawed (a wrong gold label; a comparison question with an alias). *How to say it:* "I found them with a check whose answer is known (no poison means no poisoned answers), documented them, and Block F measures their influence separately."

### Paper · Introduction (G1.2)
- **What it says:** RAG makes models answer from retrieved text, so a RAG system is only as trustworthy as that text. Prior work shows models can be misled (majority rule, ClashEval's >60% adoption, PoisonedRAG's ~90% attack success). My question is narrower: can a simple instruction protect an open model, and does it notice the conflict?
- **How to explain it to Prof. Ferrara:** "This is not a new method. I built a controlled setting where I decide exactly how many passages lie and how, and measured three open models under three prompts that differ by one sentence. All rules and tests were fixed before the main run."
- **Likely question:** "Isn't this already known?" **Answer:** "That models can be misled, yes, and I cite it. What I measure is whether a prompt-level instruction helps, at what cost on clean evidence, and whether noticing a conflict goes with resisting it."
- **Limitation to say first:** small open models and synthetic poisons, so a controlled stress test, not everyday conditions.