# 1. Tokens, probability, and representations

**Question:** What object is a language model predicting, and how can a numerical function accept text symbols?

This page follows Breeden §§1–3. The probability factorization is exact; the choice of tokenizer and representation is a modeling design.

## Tokens are discrete inputs

A tokenizer maps text to IDs in a finite vocabulary \(V\). A token can be a word, word fragment, punctuation mark, whitespace marker, or special symbol. For IDs \(w_1,\ldots,w_n\), the chain rule says

\[
P(w_{1:n})=\prod_{t=1}^{n}P(w_t\mid w_{1:t-1})
=\prod_{t=1}^{n}P(w_t\mid w_{<t}).
\]

**Intuition:** predict one continuation at a time, then multiply the conditional probabilities. **Exact fact:** this is a probability identity, not a Transformer architecture. The model supplies the conditional distributions.

For a next-token target at position \(t\), the input is usually \(w_{<t}\) and the target is \(w_t\). During training, all positions can be evaluated in parallel because the causal mask prevents position \(t\) from seeing its target or later tokens.

## Why share parameters?

There are \(|V|^m\) possible length-\(m\) contexts. A lookup table with a separate distribution for every context grows exponentially in \(m\), shares nothing between related contexts, and cannot gracefully represent unseen contexts. A neural model instead applies the same fitted parameters to many contexts. Parameter sharing is the reason a function can generalize; it is not a guarantee that it will generalize well.

## Embedding lookup

Let \(E\in\mathbb R^{|V|\times d}\) be a learned table. Token ID \(w_i\) selects one row:

\[
x_i=E[w_i]\in\mathbb R^d.
\]

The operation computes a row selection. The model uses it because matrix operations need numerical vectors rather than arbitrary integer labels. For example, with

\[
E=\begin{bmatrix}1&0&0\\0&1&0\\1&1&0\end{bmatrix},
\quad w_i=2,
\]

the lookup is \(x_i=(1,1,0)\), a vector in \(\mathbb R^3\). It is not multiplication by the number 2.

### Embeddings are not PCA

**Technical clarification:** Breeden uses PCA as a compression analogy (source §3.2). Standard embeddings are not generally computed by running PCA over token counts or hidden states. They are learned parameters updated through the prediction objective. Their coordinates need not be orthogonal, ordered by variance, or individually interpretable. Similar usage can produce nearby vectors, but similarity is a learned consequence, not a constraint.

### Lookup versus contextual state

| Object | Shape | Context-dependent? |
|---|---:|---:|
| `E[w_i]` | \(d\) | No |
| layer state at position \(i\) | \(d\) | Yes, after mixing |

The two occurrences of “bank” receive the same initial row but can acquire different states in “river bank” and “bank account.” Contextualization is performed by later operations, especially attention; the lookup alone cannot resolve the sense.

## Go connection

`Model.Embeddings` in `go/minillm/model.go` has shape `[vocabulary, width]`. In `Forward`, `m.Embeddings[id]` selects a row and `sinusoidalPosition(i, width)` supplies a separate position vector before `add` combines them. The shape check rejects token IDs outside the table.

## Common misconceptions

- The chain rule does not imply a neural network; any valid conditional distribution can use it.
- An integer token ID is a category, not a meaningful scalar distance.
- A contextual state is not the same object as the static embedding row.
- An embedding coordinate is not automatically a human concept or a PCA component.

## Self-check

1. For a vocabulary of 50,000 and width 768, what are the dimensions of \(E\)?
2. Which operation first lets the state for “bank” depend on “river”?
3. Why does the chain rule remain true even if the model's probabilities are poorly fitted?

<small>Source: Breeden §§1–3. Added technical reference: Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), §3.4.</small>
