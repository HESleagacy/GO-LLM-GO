# 6. Limits and interpretive claims

**Question:** Which statements follow from the equations, which report observations, and which are the author's interpretation?

This page follows Breeden §§11–13. The labels are deliberate because a mechanism description and a capability claim are different kinds of evidence.

## Three claim categories

### Mathematical consequence

Under the stated assumptions, the equations establish that:

- a causal attention row has zero weight on positions \(j>i\);
- softmax outputs non-negative values summing to one, for finite logits;
- the chain rule factors a sequence probability into conditionals;
- an output projection produces one real score per vocabulary item.

These are properties of the computation, independent of whether the parameters are useful.

### Empirical finding

An empirical claim needs a trained model, a dataset, a measurement procedure, and a scope. For example, a study might find that a head's attention pattern correlates with a syntactic dependency on a benchmark. That finding does not imply every model or head has the same pattern. The equations alone cannot establish it.

### Author interpretation

Breeden argues that next-token prediction can yield behavior that looks like reasoning while distinguishing it from formal symbolic reasoning. This is a useful framing of the paper, but it is not a theorem derived from softmax or matrix multiplication. The right question for a capability claim is: what behavioral test, baseline, and trained checkpoint support it?

## What the objective does and does not say

The direct objective is next-token likelihood. It does not explicitly name truth, grammar, grounding, reasoning, or human concepts. Those structures may be useful for reducing prediction loss and may emerge in a particular model, but the objective does not prescribe a unique internal explanation.

## Limitations

- **Grounding:** a text-only training objective receives text, not direct physical-world measurements. Multimodal systems change this premise.
- **Hallucination:** high probability is not the same as truth, and likelihood is not automatically calibrated confidence.
- **Verification:** generated mathematics or code is not validated merely because it was likely. An external checker is needed when correctness matters.
- **Context:** a fixed-window decoder directly processes a bounded prefix at each step. Retrieval, recurrence, memory, or tools can change the information supplied to the model, but they do not make the basic decoder equation say more than it does.
- **Interpretability:** attention weights are intermediate coefficients, not a complete causal account of the output.
- **Scaling claims:** this repository contains no benchmark, trained checkpoint, or quality evaluation. No performance claim should be inferred from the Go demo.

## How to benchmark the effect of language

**Question:** Does the same model behave differently when an equivalent task is written in different natural languages?

Do not compare unrelated prompts and call the difference a language effect. Use a paired, preregistered evaluation:

1. Select a task set with license, source, and contamination checks recorded. Include the same underlying examples in each language, with native-speaker translation or independently written matched items.
2. Keep the model checkpoint, tokenizer, decoding settings, maximum output length, system instructions, and number of examples fixed. Record whether the tokenizer splits each language into different numbers of tokens; token counts affect context pressure and cost.
3. Evaluate at least three dimensions separately: task quality (for example exact match, pass rate, or human-rated rubric), calibration (negative log-likelihood or accuracy when probabilities are available), and efficiency (input/output tokens, latency, and memory). Do not treat raw generated text length as quality.
4. Use deterministic greedy decoding for the primary comparison. If sampling is relevant, run multiple fixed seeds and report the mean and a confidence interval, not the best run.
5. Report per-language results, paired differences on the same examples, sample counts, missing or invalid outputs, confidence intervals, and the evaluation script or hashes. Include a strong translation or language-agnostic baseline where possible.
6. Inspect failure categories: factuality, reasoning step, instruction following, formatting, code-switching, named entities, and tokenizer/context overflow. Aggregate scores can hide a language-specific failure mode.

Interpret the result carefully. A lower score may reflect data coverage, translation quality, cultural assumptions, tokenizer fragmentation, script handling, or benchmark leakage rather than an intrinsic property of the language. Compare the model with a human or task-specific baseline when the task permits it. This repository provides no trained checkpoint or multilingual tokenizer, so this is a measurement plan, not an experiment already performed.

## Corrections retained in this guide

The source is valuable as an accessible derivation, but several broad statements need qualifiers:

1. Embeddings are learned lookup parameters, not generally PCA.
2. Multiple heads allow different projections but do not guarantee semantic specialization or form a mathematical basis.
3. Residual paths provide a direct derivative route but do not guarantee non-vanishing gradients.
4. LayerNorm, RMSNorm, pre-norm, and post-norm are distinct design choices.
5. Stochastic gradient methods do not universally guarantee global convergence on deep non-convex objectives.
6. A fixed five-token, fixed-weight forward pass is not a trained SLM or an inference-ready model.

## Self-check

1. Is “future positions receive zero attention weight” a mathematical consequence or an empirical finding?
2. What evidence would be required before saying a particular head detects subjects?
3. Why can a model assign high probability to a false sentence?

<small>Source: Breeden §§11–13. Added caution follows standard distinctions in Transformer interpretability and empirical evaluation; no capability or benchmark claim is made here.</small>
