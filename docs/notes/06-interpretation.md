# 6. What models learn—and what the math alone cannot settle

This page follows source sections 11–13. It separates the model's objective and operations from interpretations of learned behavior.

## 6.1 The training objective is narrow and precise

The direct objective described in the paper is next-token likelihood. Parameters are adjusted so observed continuations receive higher probability. The objective does not explicitly ask the model to learn grammar, truth, world models, or human concepts. Such structure may be useful for prediction and may emerge in trained representations, but the objective does not prescribe a unique internal explanation.

## 6.2 Interpretable patterns are evidence, not assigned jobs

Analyses of trained models have found attention patterns and internal features that correlate with linguistic relationships. The paper emphasizes that parallel heads are not manually assigned roles. That warning is useful. However, any claim that a particular head represents a clean linguistic relation requires empirical analysis of a particular model and method; it does not follow from the Transformer equations.

## 6.3 Capability claims need scope

The paper argues that next-token prediction can produce capabilities that look like reasoning, while denying that this is formal symbolic reasoning. Treat this as the author's interpretation, not a theorem established by the preceding equations. Behavioral evidence varies by task and model. The safe mathematical claim is narrower: a trained model approximates conditional distributions over token sequences; what internal computation best explains a behavior requires separate evidence.

## 6.4 Limitations in the paper's list

- **Grounding:** A text-only model learns statistical structure from text and does not directly observe the physical world through that training objective. Multimodal models alter this premise.
- **Hallucination:** High likelihood under a learned distribution is not the same as factual truth or calibrated confidence.
- **Formal verification:** Neural generation does not itself guarantee a proof is valid. External checkers can validate formal objects when the task can be expressed in their language.
- **Context window:** A standard fixed-window decoder directly processes a bounded prefix at each step. Retrieval, recurrence, memory, or other mechanisms can change what information is available over a task.

## 6.5 Corrections and nuances worth retaining

The source is useful as an accessible derivation, but several sentences overstate what the equations prove. The [source notes](../source-notes.md) identify the main ones. In particular, embeddings are not generally PCA; multiple heads need not specialize; normalization and residual layouts vary; and stochastic gradient descent does not universally “converge” in the simple sense implied by an informal sentence.

## Conclusion in plain language

A Transformer is a parameterized function that repeatedly mixes token representations according to learned compatibility scores, transforms them nonlinearly, and outputs next-token probabilities. The ingredients are familiar mathematics. The learned parameters and scale make the behavior complicated. That gives us a tractable computational description, not an automatic explanation of every capability or a guarantee of truth.
