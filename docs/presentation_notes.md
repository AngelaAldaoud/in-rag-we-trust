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