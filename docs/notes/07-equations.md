# 7. Equation reference

This page collects the central equations in consistent notation. Indices are 0-based in code and 1-based in the paper; the math is unchanged.

## Sequence probability

\[
P(w_{1:n})=\prod_{t=1}^{n}P(w_t\mid w_{<t}).
\]

## Embedding and position

\[
x_i^{(0)}=E[w_i]+p_i.
\]

Here \(p_i\) is one possible absolute positional representation. Other schemes modify attention scores or projections instead.

## Single-head causal self-attention

\[
q_i=W_Qx_i,\qquad k_j=W_Kx_j,\qquad v_j=W_Vx_j.
\]

\[
s_{ij}=\begin{cases}q_i^\top k_j/\sqrt{d_k},&j\le i,\\-\infty,&j>i,\end{cases}
\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{r\le i}e^{s_{ir}}},
\qquad
z_i=\sum_{j\le i}\alpha_{ij}v_j.
\]

Shapes: \(x_i\in\mathbb R^d\), \(q_i,k_i\in\mathbb R^{d_k}\), \(v_i,z_i\in\mathbb R^{d_v}\), \(W_Q,W_K\in\mathbb R^{d_k\times d}\), and \(W_V\in\mathbb R^{d_v\times d}\).

## Multi-head combination

\[
z_i^{(h)}=\operatorname{Attention}^{(h)}(x_{1:n})_i,
\qquad y_i=W_O[z_i^{(1)};\cdots;z_i^{(H)}].
\]

## Residual and feed-forward block (schematic)

\[
r_i=x_i+y_i,\qquad x_i'=r_i+W_2\phi(W_1r_i+b_1)+b_2.
\]

Normalization is omitted here because its location and form vary by architecture.

## Vocabulary distribution

\[
z_v=u_v^\top x_t^{(L)}+b_v.
\]

\[
P(w_{t+1}=v\mid w_{1:t})=\frac{e^{z_v}}{\sum_{r\in V}e^{z_r}}.
\]

## Training objective

\[
\mathcal L(\theta)=-\sum_{t=1}^{N}\log P_\theta(w_t\mid w_{<t}),
\qquad
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta).
\]

In practice the loss may be averaged, masked over padding, combined across batches, and optimized with methods beyond plain gradient descent.
