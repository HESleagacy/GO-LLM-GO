# Source, scope, and technical notes

## Source document

Joseph L. Breeden, *The Simple Mathematics of Large Language Models*, 20-page PDF supplied in this conversation. The PDF cover says January 2026; its title page says “9 December 2025.” The notes cite section numbers because the PDF does not supply a DOI or stable publication URL. Its references are reproduced in the PDF itself.

This guide is a study aid based on that source, not a replacement edition. It paraphrases rather than reproduces the paper wholesale. The explanations, examples, and Go code are newly written for this guide.

## Source-grounded claim versus added clarification

The section notes track the author's exposition and preserve his emphasis on ordinary probability, linear algebra, and optimization. Technical qualifications are explicitly introduced as notes; they should not be mistaken for quotations or claims made by the author.

## Important qualifications

1. **Embeddings and PCA (source §3.2):** a token embedding is a learned lookup table. Calling it “like PCA” may suggest a statistical procedure that is not generally used to obtain it. Treat the analogy as loose compression intuition only.
2. **Finite context (source §2.3):** a fixed-context decoder's next-token computation is bounded by its available window. Calling every LLM a finite-order Markov model “in a precise sense” needs the model class and context mechanism specified; retrieval, recurrent state, or memory changes the effective dependency structure.
3. **Head specialization (source §6):** separate heads permit different projections. The equations do not force distinct linguistic jobs, and heads can be redundant. “Implicit pressure to differentiate” is an intuition, not a theorem.
4. **Basis terminology (source §6):** calling heads “basis elements” is metaphorical unless independence and spanning properties are established. The attention outputs are learned functions; they are not automatically a mathematical basis.
5. **Residual gradients (source §7.2):** the identity path contributes a direct derivative route, which often helps optimization. It does not guarantee non-vanishing gradients throughout a network.
6. **Normalization (source §7.3):** Transformer variants differ in normalization type and placement. LayerNorm, RMSNorm, pre-norm, and post-norm are not interchangeable descriptions of a single fixed rule.
7. **Optimization (source §10.4):** minibatch gradients can be unbiased under sampling assumptions. This does not guarantee convergence to a global optimum for a deep non-convex model.
8. **Reasoning and grounding (source §11):** strong categorical statements about what models can or cannot reason are interpretive and task-dependent. The equations establish a predictive computation, not a complete theory of cognition.
9. **Decoder masking:** the source's opening explanation speaks of the whole context; its mathematical summary restricts aggregation to \(j\le i\). This guide makes the causal mask explicit wherever next-token prediction is implemented.

## Implementation scope

The Go code demonstrates a dependency-free CPU forward pass for a tiny one-head causal model. Its parameters are deterministic toy values. It does not train, tokenize raw text, use RoPE, reproduce any paper experiment, or approximate a frontier model's quality. The limitations are stated prominently to prevent an educational demo from being mistaken for a trained LLM.

## Further reading listed by the source

The source PDF includes references to Vaswani et al. (2017), *Attention Is All You Need*; Su et al. (2024), *RoFormer: Enhanced Transformer with Rotary Position Embedding*; Press et al. (2022), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*; and the other works listed in its final pages. Use the source PDF's bibliography for complete citation details.
