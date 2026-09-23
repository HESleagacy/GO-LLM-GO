# Go Lab: inspect a causal forward pass

This lab turns the central equations into ordinary Go slices and loops. It is intentionally small enough to audit without a tensor framework.

## Implemented path

For token IDs \(w_0,\ldots,w_{n-1}\), `Model.Forward` does the following:

1. `m.Embeddings[id]` selects a row of shape `[width]`; `sinusoidalPosition` creates an absolute position vector of the same shape; `add` forms the state.
2. `MatVec` applies `WQ`, `WK`, and `WV` at every position, producing Q, K, and V rows.
3. `CausalAttention` computes only scores for `j <= i`, divides by `sqrt(keyWidth)`, calls stable `Softmax`, and sums visible value rows.
4. `WO` projects the attention result and an `add` applies the attention residual.
5. `W1`, `relu`, and `W2` implement the position-wise feed-forward path; another `add` applies its residual.
6. `Output` maps the final position to vocabulary logits; `Softmax` returns the next-token distribution.

`ForwardStates` exposes the final state for each position without applying the vocabulary classifier. It exists to make the causal invariant observable in a test; it is not a KV cache and does not change `Forward`'s output.

This is one head and one unnormalized block-shaped pass. It has the broad decoder computation, but not the full range of production Transformer variants.

## Run it

From the project root:

```sh
cd go
go test ./...
go vet ./...
go run ./cmd/demo
```

The demo prints five probabilities whose sum is close to `1.000000`. The numbers are deterministic smoke-test output from fixed toy weights. They are not language predictions, benchmark results, or a trained SLM.

## Attention invariant

`CausalAttention` accepts `q`, `k`, and `v` matrices with shapes `[sequence, keyWidth]`, `[sequence, keyWidth]`, and `[sequence, valueWidth]`. For each row `i`, it materializes a score vector of length `i+1`; future positions are never passed to softmax. `TestCausalAttentionDoesNotReadFutureValues` changes only the value at position 2 and verifies positions 0 and 1 are unchanged while position 2 can change.

The code uses

\[
\operatorname{softmax}(s)_j=\frac{\exp(s_j-m)}{\sum_r\exp(s_r-m)},
\qquad m=\max_r s_r,
\]

which is algebraically equal to ordinary softmax and avoids overflow. `TestSoftmaxStableAndNormalized` checks finite values and a sum within \(10^{-12}\) of one.

## Math-to-code map

| Mathematics | Go implementation | Test / check |
|---|---|---|
| \(E[w_i]+p_i\) | `model.go`, `Forward` | token ID bounds |
| \(W_Qx_i,W_Kx_i,W_Vx_i\) | `MatVec` calls in `Forward` | rectangular shapes |
| masked \(q_i^\top k_j/\sqrt{d_k}\) | `attention.go`, `CausalAttention` | future-value invariance |
| stable softmax | `math.go`, `Softmax` | large-logit test |
| \(W_2\operatorname{ReLU}(W_1x)\) | `relu` and `MatVec` calls | feed-forward shapes |
| \(Uh_t\) and vocabulary probabilities | final `MatVec` and `Softmax` | normalized output |

## What is not implemented

- tokenizer, text normalization, or ID vocabulary file;
- gradient calculation, backpropagation, optimizer, or training data;
- batching, checkpoint save/load, GPU kernels, or optimized tensor storage;
- configurable multi-head/multi-layer architecture or normalization;
- RoPE, decoding controls, generation loop, or KV cache.

The fixed `DemoModel` parameters are selected for reproducibility. Structurally valid probabilities do not imply semantic validity. The [inference-readiness roadmap](inference-readiness.md) lists the concrete work required before this could load a trained checkpoint and generate text.

## Suggested exercises

1. Add a function that returns attention weights for inspection, then test every row sums to one.
2. Add a second head with a different value width and concatenate outputs before `WO`.
3. Add a normalization function, explicitly choose pre-norm or post-norm, and test its numerical behavior.
4. Implement cached decoding only after an uncached generation loop exists; compare logits within a stated tolerance.
