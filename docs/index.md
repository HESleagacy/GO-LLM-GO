# The Mathematics of LLMs, worked through in Go

This guide unpacks the uploaded paper *The Simple Mathematics of Large Language Models* by Joseph L. Breeden. It follows the paper's path from discrete tokens to conditional probabilities, then adds small Go implementations that make the central operations concrete.

The running model is deliberately tiny. Its job is to expose the mechanics: vectors, projections, causal masking, weighted sums, residual paths, and output probabilities. It is not a trained or production language model.

## Start here

1. Read the [Master Notes](master-notes.md) for the paper's argument and navigation.
2. Work through the [concept notes](notes/01-foundations.md) in order.
3. Run the [Go Lab](go-lab.md) and inspect each operation against its equation.
4. Use the [equation reference](notes/07-equations.md) for a compact recap.
5. Use [Inference readiness](inference-readiness.md) as the checklist for turning the demo into a small, usable model.
6. Read [Source and scope](source-notes.md) for corrections and claims that need qualification.

## What this guide teaches

- A language model estimates the next-token conditional distribution.
- Learned embeddings turn token IDs into vectors that can be transformed.
- Self-attention forms a weighted sum of projected token representations.
- A causal mask prevents a decoder from using future tokens.
- Stacking attention and nonlinear transformations creates deeper context-dependent features.
- Training adjusts parameters to reduce next-token negative log-likelihood.

## What it does not pretend

The Go program is a forward-pass demonstration with deterministic toy weights. It does not contain tokenizer training, automatic differentiation, optimizer code, data loading, GPU kernels, or pretrained parameters. A handful of hand-set numbers can demonstrate equations; they cannot conjure a competent LLM. See [the implementation boundary](go-lab.md#scope-of-the-go-implementation).

## Source

All paper-specific explanations are grounded in Breeden's uploaded 20-page PDF. Additional notes are labeled as **Correction**, **Nuance**, or **Implementation note** so the paper's statements are not silently mixed with general technical clarification.
