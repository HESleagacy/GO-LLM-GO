# Go Lab: a tiny causal language-model forward pass

This lab turns the core equations into ordinary Go slices and loops. The package has no third-party dependencies.

## Model path

For token IDs \(w_0,\ldots,w_{n-1}\):

1. Look up a learned-looking (but hand-set) embedding row and add a sinusoidal position vector.
2. Apply query, key, and value matrices.
3. Compute scaled dot products, mask all future positions, softmax the visible scores, and mix value vectors.
4. Apply an output projection and a residual addition.
5. Apply a ReLU feed-forward network and another residual addition.
6. Project the last position to vocabulary logits and compute stable softmax probabilities.

The mathematical structure is the same as the conceptual decoder in the notes. The implementation uses one head and omits normalization, biases in most projections, batching, and training so each operation stays inspectable.

## Run it

From the project root:

```sh
cd go
go test ./...
go run ./cmd/demo
```

Expected output includes a probability for each toy vocabulary token and a sum close to `1.000000`. The exact probabilities are only a deterministic smoke-test output; they are not a learned language model.

## The core computation

`CausalAttention` receives projected query, key, and value rows. For each position `i`, it computes scores only for `j <= i`, divides dot products by `sqrt(d_k)`, applies stable softmax, then returns the weighted sum of value rows. This directly implements the equation in [Context and weighted averaging](notes/02-context-attention.md).

The code uses the mathematically equivalent stable softmax form:

\[
\operatorname{softmax}(s)_j=\frac{\exp(s_j-m)}{\sum_r\exp(s_r-m)},\quad m=\max_r s_r.
\]

Subtracting the same maximum from every score avoids overflow without changing the result.

## Tests and what they establish

- Softmax returns finite values summing to one even for large logits.
- A future value vector cannot affect the attention output at an earlier position.
- Invalid shapes and token IDs return errors instead of panicking.
- The model returns a normalized distribution over its toy vocabulary.

These test mathematical invariants of the demo. They do **not** establish that a language model is well-trained or linguistically capable.

## Scope of the Go implementation

This is a forward-pass teaching model, not a full LLM implementation. It does not include:

- text normalization or a tokenizer;
- gradient calculation, backpropagation, or parameter updates;
- a corpus loader, batching, checkpointing, GPU execution, or efficient tensor kernels;
- multi-head or multi-layer configuration;
- RoPE, KV caching, sampling strategies, or a pretrained checkpoint.

The parameters in `DemoModel` are fixed, small values selected to make execution reproducible. Thus the output probabilities are structurally real but semantically meaningless. A credible training implementation would need a tokenizer, data pipeline, autodiff, optimizer, validation, and enough compute/data; pretending a 100-line demo did all that would be theatre.

For the concrete work required to cross that gap, see [Inference readiness](inference-readiness.md).

## Suggested extensions

1. Add a second attention head and concatenate its output before `WO`.
2. Add a normalization layer and compare pre-norm with post-norm.
3. Implement scalar reverse-mode autodiff, then use it to train a tiny next-token model.
4. Replace sinusoidal positions with RoPE and test how relative displacement affects scores.
5. Add a real tokenizer and a tiny corpus only after the forward pass and gradients are independently verified.
