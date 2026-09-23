# 4. Position and next-token probabilities

This page follows source sections 8–9: give the model information about order, then map its final state to a vocabulary distribution.

## 4.1 Why order must enter

Without position information, self-attention sees a collection of vectors and is permutation-equivariant: reorder the input vectors and the outputs reorder correspondingly. It has no intrinsic signal that “dog bites man” differs from “man bites dog.” Position must be encoded in the input or attention calculation.

## 4.2 Position methods

The paper contrasts absolute encodings, which add a vector \(p_i\) to the token embedding, with relative-position methods. Rotary Position Embedding (RoPE) rotates query and key coordinates by position-dependent angles so their dot products depend on relative displacement. ALiBi instead adds a distance-dependent bias to attention scores.

These are different mechanisms; neither is “just another embedding table.” The source's conceptual point is sound: without some position signal, order-sensitive next-token modeling cannot work.

## 4.3 From final state to vocabulary scores

Given final state \(x_t^{(L)}\), a linear output map gives one score per token:

\[
z_v=u_v^\top x_t^{(L)}+b_v.
\]

Softmax converts scores to probabilities:

\[
P(w_{t+1}=v\mid w_{1:t})=\frac{\exp(z_v)}{\sum_{r\in V}\exp(z_r)}.
\]

The vector \(u_v\) is the output classifier vector for token \(v\). Some implementations use an output matrix and tie it to the embedding matrix; the paper's equation shows separate learned vectors.

## 4.4 Scores are not probabilities

Logits can be any real numbers. Softmax makes them non-negative and sums them to one. Adding the same constant to every logit changes no probability. This is why stable softmax can subtract the largest logit first:

\[
\operatorname{softmax}(z)_v=\frac{\exp(z_v-m)}{\sum_r\exp(z_r-m)},\qquad m=\max_r z_r.
\]

The model's top-probability token is the most likely next token according to its learned distribution, not necessarily the best or true continuation in an external sense.

## Go connection

The tiny model adds sinusoidal absolute position vectors and applies a vocabulary projection. The position method is intentionally simple and inspectable. It is not an implementation of RoPE; see [Source and scope](../source-notes.md) for why the demo chooses a simpler route.
