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