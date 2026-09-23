# Notation and dimensions

Use this page when an equation feels ambiguous. Vectors are column vectors in the equations, while the Go package stores a vector as a row-shaped `[]float64` and applies `MatVec(M, x)` to compute the same mathematical product.

## Core symbols

| Symbol | Meaning | Shape in one sequence |
|---|---|---|
| \(V\) | vocabulary set | \(|V|\) token types |
| \(n\) | sequence length | scalar |
| \(d\) | model width | scalar |
| \(d_k\) | query/key width | scalar |
| \(d_v\) | value/head width | scalar |
| \(E\) | token embedding table | \(|V|\times d\) |
| \(X\) | states for all positions | \(n\times d\) |
| \(W_Q,W_K\) | query/key projections | \(d_k\times d\) |
| \(W_V\) | value projection | \(d_v\times d\) |
| \(Q,K\) | projected queries/keys | \(n\times d_k\) |
| \(V_{m val}\) | projected values | \(n\times d_v\) |
| \(S\) | attention scores | \(n\times n\) |
| \(A\) | attention weights | \(n\times n\) |
| \(Z\) | attention output | \(n\times d_v\) |
| \(W_O\) | multi-head output projection | \(d\times(Hd_v)\) |
| \(U\) | vocabulary output matrix | \(|V|\times d\) |
| \(z\) | vocabulary logits | \(|V|\) |

For a single position, \(x_i\in\mathbb R^d\), \(q_i,k_i\in\mathbb R^{d_k}\), and \(v_i,z_i\in\mathbb R^{d_v}\). The symbol \(z_i\) is used for an attention output in some pages; vocabulary logits are also often called \(z\). Context makes the distinction clear, and the code calls the attention result `context`.

## Shape checks by operation

1. Embedding lookup: `E[w_i]` selects one row, so \((|V|\times d)\to(d)\).
2. Projection: \((d_k\times d)(d)\to(d_k)\), and similarly for \(W_K,W_V\).
3. Score: \(q_i^\top k_j\) multiplies \((1\times d_k)(d_k\times1)\to(1)\).
4. Weighted sum: \(\sum_j\alpha_{ij}v_j\) stays in \(\mathbb R^{d_v}\).
5. Output classifier: \((|V|\times d)(d)\to(|V|)\).

!!! note "Code convention"
    `Matrix` is row-major. In `Model`, `Embeddings` has shape `[vocabulary, width]`, each `WQ`, `WK`, `WV`, and `WO` is `[width, width]`, `W1` is `[hidden, width]`, `W2` is `[width, hidden]`, and `Output` is `[vocabulary, width]`. These are the deliberate restrictions of the teaching model, not the only valid Transformer shapes.

## Index convention

The source paper writes positions as \(1,\ldots,n\); Go slices use \(0,\ldots,n-1\). In either convention “future” means a key position strictly greater than the query position. Thus the causal condition is \(j\le i\).

## References

- Breeden, *The Simple Mathematics of Large Language Models*, source sections 2–9 (supplied PDF; not present in this checkout).
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017, equations 1–3.
