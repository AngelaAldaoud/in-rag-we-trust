# Results summary (Block F)

Plain-English findings of "In RAG We Trust?". Each points to its figure (`results/figures/`), table (`results/analysis/tables/`) and macros (`results/analysis/numbers.tex`, `\X`). Confirmatory = planned tests of the frozen analysis plan; everything else is exploratory and labelled. Numbers in this file are rounded for reading; the paper uses the macros.

## Confirmatory findings

**1. More false passages, more false answers (RQ1, H1: partly supported).**
As the share of poisoned passages rho grows from 0 to 1, the poisoned-answer rate (ASR) rises from about 0% to `\AsrRangeRhoOneMin`-`\AsrRangeRhoOneMax`% (about 55-58%) and accuracy falls from `\AccRangeRhoZeroMin`-`\AccRangeRhoZeroMax`% to `\AccRangeRhoOneMin`-`\AccRangeRhoOneMax`%, for all three models. The odds ratio from rho=0 to 1 is 50-65 (`\FOneOr{Llama,Mistral,Qwen}`), Holm p<0.001.
Fig 1 · T1. Only partly supported because H1 called rho=0.5 a majority; with four passages it is a 2-2 tie (D262). The largest ASR step is 0.5 -> 0.75 for Mistral and Qwen, when the poison becomes the actual majority, but 0.75 -> 1 for Llama (`\StepMax*`, `\StepFrom*`, `\StepTo*`).

**2. Meta-prompts help a little, mostly by refusing (RQ2, H2: partly supported).**
Asking the model to check consistency or source reliability lowers ASR at rho>0 in `\FTwoAsrSig` of `\FTwoAsrTests` comparisons (about 2-8 points; Mistral-reliability largest, Qwen-reliability not significant). On clean evidence it raises abstention (`\FTwoAbstUp` of `\FTwoAbstTests`) and costs accuracy significantly for Mistral only.
Fig 2 · T1 · `\FTwoAsr*`, `\FTwoAbst*`, `\FTwoAcc*`.

**3. Flagged conflicts go with fewer poisoned answers (RQ3, H3: supported, association only).**
When the judge flags a conflict, the odds of a poisoned answer are lower (OR about 0.81-0.84, `\FThreeOr*`) for all three models. Flags rise most above each model's clean baseline (`\FlagBase*`) at rho=0.5, where the evidence is mixed (`\FlagDiff*RhoHalf`), and less at rho=1 (`\FlagDiff*RhoOne`).
Fig 3 · T1. This is an association: flagging is not shown to cause the lower ASR.

**4. Poison types barely differ; models do (RQ4, H4: partly supported).**
All `\FFourSig` interaction blocks are significant, but the effect is small. Pooled over rho>0, the four poison types differ by at most `\TypeSpreadPooledMax` points within a model (`\TypeSpreadPooled*`); at a single rho the largest gap is `\TypeSpreadMax` points (`\TypeSpread*`); in neither case is there a consistent order of types. Llama is poisoned less often than Mistral and Qwen (`\TypeAsrPooled*`).
Fig 4 · T1 · `\FFourChi*`.

**Verdicts:** T2 (D250, D263).

## Exploratory findings

- **False context overrides what the model knows.** At rho=1, accuracy is far below closed-book accuracy (`\ClosedAccMin`-`\ClosedAccMax`%), and questions answered correctly without passages are still poisoned `\PriorGoldAsrRhoOneMin`-`\PriorGoldAsrRhoOneMax`% of the time at rho=1. Fig 1 · `\PriorAsr*`.
- **Crossover.** The first tested rho with ASR above Acc is 0.75 for every model under the rules; under the judge-based upper bound it moves to 1.0 for Llama. `\CrossRules*`, `\CrossUpper*`, `\CrossCurvesAtThreeQuarters` of `\CrossCurves` curves.
- **How the meta-prompts work.** `\TransStayMin`-`\TransStayMax`% of plain-poisoned answers stay poisoned. Abstention is the main exit for Mistral and Qwen; Llama's exits split between abstain, correct and other. Fig 5 · `\Trans*`.
- **Scoring bracket.** The judge-based upper bound adds `\BrAccGapMin`-`\BrAccGapMax` points to accuracy and at most `\BrAsrGapMax` to ASR. `\BrAcc*`, `\BrAccPlus*`.
- **NoAnswer and hallucination.** NoAnswer is highest for Llama (`\NoAnswerLlama`%, mostly length-cap hits, `\NoAnswerTruncLlama`%). Hallucinations with passages are rare (`\HallMin`-`\HallMax`%), detector precision `\HallPrecision`, miss rate `\HallMiss`.
- **Retrieval.** With BM25 retrieval, ASR follows the effective share of poisoned passages. `\RetrAsr*`, `\RetrShare*`.
- **Position.** At rho=0.25, Qwen is poisoned more often when the single poisoned passage comes first. `\PosAsr*`.
- **Leaks.** On the `\LeakQuestions` questions whose poison leaks the gold answer, ASR is lower. `\LeakDiff*`.
- **Power.** The median minimum detectable difference per cell is `\MddMedian` points; `\MddUnderpowered` of `\MddCells` cells have below 80% power for a 5-point difference, so non-significant single cells are not evidence of no effect.

## Robustness (exploratory)

The audit flagged `\AuditFlagged` of `\AuditCandidates` candidate questions (`\AuditGoldErrors` gold errors, `\AuditComparison` comparison questions). Without them, `\SensFlaggedSame` of `\SensConclusions` conclusions are unchanged; without the leak questions, `\SensLeaksSame` of `\SensConclusions` (Llama-reliability ΔASR and Mistral-consistency ΔAcc become non-significant). T3.

## Limitations to state

Accuracy is a bracket (rules vs upper bound); Llama's crossover depends on the scoring; H1's majority wording is a tie; the poison types are a weak manipulation; the judge over-flags clean contexts and agrees less with humans for Qwen (kappa 0.57); flags are an association only; single cells are under-powered; comparison questions with absurd targets; leaks.