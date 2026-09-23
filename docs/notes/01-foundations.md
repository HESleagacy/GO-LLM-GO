# 1. Text, probability, and embeddings

This page follows source sections 1–3: define the prediction problem, factor sequence probabilities, and replace arbitrary token labels with learned vectors.

## 1.1 Tokens are discrete symbols

A tokenizer maps text into IDs from a finite vocabulary \(V\). A token can be a whole word, a word fragment, punctuation, or another symbol. For a sequence of IDs \((w_1,\dots,w_n)\), the chain rule gives:

\[
P(w_1,\ldots,w_n)=\prod_{t=1}^{n}P(w_t\mid w_1,\ldots,w_{t-1}).
\]

This factorization is an identity from probability, not a modeling approximation. A language model supplies the conditional distributions. A causal language model is trained to predict each next token from the preceding prefix.

## 1.2 Why store a reusable function?

With \(|V|\) token types, there are \(|V|^m\) possible contexts of length \(m\). A table that stores a separate next-token distribution for every context is therefore impractical and cannot sensibly handle unseen contexts. A model instead shares parameters across contexts and learns patterns that can generalize.

## 1.3 Token vectors

A one-hot vector gives each token a separate coordinate and does not express similarity. An embedding table \(E\in\mathbb R^{|V|\times d}\) maps token ID \(w_i\) to a row vector:

\[
x_i=E[w_i]\in\mathbb R^d.
\]

The coordinates are learned parameters. Similar usage can lead to similar representations, but the model does not require every coordinate to have a human-readable meaning.

!!! note "A useful correction to the paper's analogy"
    The paper compares embeddings to PCA as a kind of compression. Treat that only as a loose analogy: standard token embeddings are learned jointly with the model's prediction objective; they are not generally produced by running PCA. Their width \(d\) is an architecture choice, not a PCA component count.

## 1.4 Static embedding versus contextual state

The lookup \(E[w_i]\) is the same each time token ID \(w_i\) appears. The later layer states are not: after context mixing, the two occurrences of “bank” in “river bank” and “bank account” can carry different information. Keep the distinction clear:

| Object | Depends on token identity? | Depends on surrounding context? |
|---|---:|---:|
| Embedding lookup \(E[w_i]\) | Yes | No |
| Layer state \(x_i^{(\ell)}\) | Yes | Yes, after contextual layers |

## Go connection

The reference code uses a slice of `float64` values for a vector and a matrix stored as rows. In `go/minillm`, each row of the embedding table represents one token. This simple representation makes dimensions visible, though real models use optimized tensor libraries and lower-precision arithmetic.

## Check your understanding

1. Why is the chain rule not itself a language-model architecture?
2. What does the embedding table learn, and what does it not guarantee?
3. Why does a fixed token embedding fail to represent context-dependent senses by itself?
