# 7. Equation reference

Indices in the paper are 1-based; indices in Go are 0-based. Shapes use column-vector notation unless a whole sequence matrix is explicitly shown. See [Notation and dimensions](../notation.md) for the full contract.

## Sequence probability

\[
P(w_{1:n})=\prod_{t=1}^{n}P(w_t\mid w_{<t}),
\qquad
\log P(w_{1:n})=\sum_{t=1}^{n}\log P(w_t\mid w_{<t}).
\]

## Input representation

\[
x_i^{(0)}=E[w_i]+p_i,
\qquad E\in\mathbb R^{|V|\times d},\quad x_i^{(0)},p_i\in\mathbb R^d.
\]

The Go demo chooses sinusoidal absolute \(p_i\); it does not implement RoPE.

## Single-head causal self-attention

\[
q_i=W_Qx_i,\qquad k_j=W_Kx_j,\qquad v_j=W_Vx_j,
\]

\[
\widetilde s_{ij}=\begin{cases}
q_i^\top k_j/\sqrt{d_k},&j\le i,\\
-\infty,&j>i,
\end{cases}
\qquad
\alpha_{ij}=\frac{e^{\widetilde s_{ij}}}{\sum_{r\le i}e^{\widetilde s_{ir}}},
\qquad
z_i=\sum_{j\le i}\alpha_{ij}v_j.
\]

Shapes: \(x_i\in\mathbb R^d\), \(q_i,k_i\in\mathbb R^{d_k}\), \(v_i,z_i\in\mathbb R^{d_v}\), \(W_Q,W_K\in\mathbb R^{d_k\times d}\), \(W_V\in\mathbb R^{d_v\times d}\).

Stable softmax replaces every allowed score \(s_j\) with \(s_j-m\), where \(m=\max_j s_j\). Masked entries are excluded from the maximum and denominator.

## Multi-head attention

\[
Z^{(h)}=\operatorname{Attention}^{(h)}(X),
\qquad
Y=\operatorname{Concat}(Z^{(1)},\ldots,Z^{(H)})W_O.
\]

If every head has value width \(d_v\), concatenation has width \(Hd_v\). The output projection maps \(Hd_v\to d\). Separate heads are capacity, not guaranteed semantic roles.

## Decoder block variants

Chosen pre-norm schematic:

\[
x_1=x+\operatorname{Attention}(\operatorname{Norm}(x)),
\qquad
x_2=x_1+\operatorname{FFN}(\operatorname{Norm}(x_1)).
\]

Position-wise feed-forward network:

\[
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2.
\]

Post-norm instead applies `Norm` after each residual addition. RMSNorm and LayerNorm are also different normalization functions. The Go demo uses neither.

## Vocabulary distribution

\[
\ell=Uh_t+b,\qquad U\in\mathbb R^{|V|\times d},\quad \ell,b\in\mathbb R^{|V|},
\]

\[
P(w_{t+1}=v\mid w_{1:t})
=\frac{e^{\ell_v-m}}{\sum_{r\in V}e^{\ell_r-m}},
\qquad m=\max_{r\in V}\ell_r.
\]

The distribution is the model result; argmax or sampling is a separate decoding operation.

## Training objective

\[
\mathcal L(\theta)=-\frac1N\sum_{t=1}^{N}\log P_\theta(w_t\mid w_{<t}),
\qquad
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta).
\]

The Go package stops before \(\nabla_\theta\): it is forward-only.
