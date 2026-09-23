# 2. Context and learned weighted averaging

This page follows source sections 4–5. The central operation is a weighted sum whose weights depend on the current input representations.

## 2.1 From scores to a contextual representation

For a query position \(i\), let \(v_j\) be the content vector available at each context position \(j\). A weighted sum is:

\[
z_i=\sum_{j=1}^{n}\alpha_{ij}v_j,
\qquad \alpha_{ij}\ge 0,\qquad \sum_j\alpha_{ij}=1.
\]

The weights are row-specific: position \(i\) can combine the same context differently from position \(r\). A raw score \(s_{ij}\) measures compatibility; softmax makes the scores into normalized weights.

## 2.2 Three learned projections

Given input state \(x_i\), calculate:

\[
q_i=W_Qx_i,\qquad k_j=W_Kx_j,\qquad v_j=W_Vx_j.
\]

The paper writes these as \(a_i=W^Ax_i\), \(b_j=W^Bx_j\), and \(c_j=W^Cx_j\). The projections serve different computational roles:

- query \(q_i\): the features used by position \(i\) to compare context positions;
- key \(k_j\): the features from position \(j\) used in that comparison;
- value \(v_j\): the features from position \(j\) that get mixed into the output.

The names do not imply that the model has a literal database. They are conventional labels for three learned linear maps.

## 2.3 Scaled dot products and softmax

The usual score and weight are:

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}},\qquad
\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{r\in A_i}\exp(s_{ir})},\qquad
z_i=\sum_{j\in A_i}\alpha_{ij}v_j.
\]

Here \(A_i\) is the set of positions visible to position \(i\). In a bidirectional encoder it may include all positions; in a decoder-only language model it includes only \(j\le i\). The denominator must use the same allowed set.

The scaling dimension is **key width** \(d_k\), not necessarily the symbol \(k\) used by the paper. Scaling keeps dot-product magnitudes from growing with width. A numerically stable implementation subtracts the largest allowed score before exponentiating; this changes no softmax probabilities.

## 2.4 Why separate maps matter

If raw vectors were compared directly using \(x_i^\top x_j\), the score would be symmetric. Separate learned projections allow the compatibility from \(i\) to \(j\) to differ from the reverse comparison. Separating \(W_K\) and \(W_V\) also lets the model use one feature set to decide *which position matters* and another feature set to determine *what content it contributes*.

The score can be rewritten as a bilinear form:

\[
q_i^\top k_j=x_i^\top W_Q^\top W_Kx_j.
\]

This is a parameterized compatibility function. If the projected width is smaller than the model width, the induced matrix has rank at most \(d_k\). That is a property of this factorization; it does not make the learned features automatically interpretable.

## 2.5 Causality is a mask, not a hope

During next-token training, the representation at position \(i\) must not depend on later ground-truth tokens. Define:

\[
s_{ij}=\begin{cases}
q_i^\top k_j/\sqrt{d_k},&j\le i,\\
-\infty,&j>i.
\end{cases}
\]

Then softmax assigns exactly zero weight to future positions. The paper's main attention explanation describes mixing over context generally; its final summary shows \(j\le i\) for the decoder case. The Go implementation applies this causal mask explicitly.

## 2.6 What the weights do not mean

The coefficients \(\alpha_{ij}\) are useful to inspect as part of the computation. They are not, by themselves, a complete causal explanation of why a model produced a token or a reliable measure of human-style importance. Later layers, residual paths, nonlinearities, and output weights all affect the prediction.

## Tiny numerical example

Suppose two allowed value vectors are \(v_1=(1,0)\) and \(v_2=(0,2)\), with weights \((0.25,0.75)\). Then:

\[
z=0.25v_1+0.75v_2=(0.25,1.5).
\]

The output is a convex combination because the weights are non-negative and sum to one. The query/key calculation is what learns those weights; the weighted sum transports the value content.

## Go connection

`CausalAttention` computes one row at a time. It masks positions after the query position, applies stable softmax to the remaining scores, then sums the values. The tests check that a future token cannot change an earlier position's attention output.
