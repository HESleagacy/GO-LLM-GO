# End-to-end worked example

This deliberately tiny calculation follows three tokens through a minimal decoder path. It is designed to be checked with a calculator, not to be a realistic trained model. The zero query/key matrices isolate the mechanics of causal visibility and weighted aggregation; the example therefore does not claim that a useful model would learn uniform attention.

## Setup

Let the vocabulary be \(V=\{A,B,C\}\), the sequence be \((A,B,C)\), and let \(n=3\), \(d=d_k=d_v=2\). Use these embedding rows and absolute position vectors:

\[
E[A]=(1,0),\quad E[B]=(0,1),\quad E[C]=(1,1)
\]

\[
p_0=(0,0),\quad p_1=(1,0),\quad p_2=(0,1).
\]

The input state matrix \(X=E[w_i]+p_i\), with shape \(3\times2\), is therefore

\[
X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}.
\]

Use one attention head with

\[
W_Q=W_K=\begin{bmatrix}0&0\\0&0\end{bmatrix},
\qquad
W_V=I_2.
\]

Thus \(Q=K=0\) and \(V_{\rm val}=X\). Every allowed score is

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{2}}=0.
\]

The mask still matters: position 0 sees one state, position 1 sees two, and position 2 sees three.

The visibility figure shows a five-position template; this three-token example uses its upper-left \(3\times3\) block.

![Causal visibility used by the example](assets/images/causal-visibility.svg){ .diagram }

<p class="diagram-caption">Figure 1. The green lower triangle is the allowed set \(j\le i\); red cells are future tokens and cannot affect earlier outputs.</p>

## Attention output

Stable softmax of a row of all zeros is uniform. The attention weights are therefore

\[
A=\begin{bmatrix}
1&0&0\\
1/2&1/2&0\\
1/3&1/3&1/3
\end{bmatrix}.
\]

Multiplying \(A V_{\rm val}\) gives the contextual states:

\[
Z=\begin{bmatrix}
1&0\\
1&1/2\\
1&1
\end{bmatrix}.
\]

For example, the final row is

\[
z_2=\tfrac13(1,0)+\tfrac13(1,1)+\tfrac13(1,2)=(1,1).
\]

No future state appears in that sum. If a fourth token were appended, the first three rows would remain unchanged under this causal attention calculation.

## Logits and probabilities

Use an output matrix with one row per vocabulary item:

\[
U=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},
\qquad
\ell=Uz_2=\begin{bmatrix}1\\1\\2\end{bmatrix}.
\]

The logits are not probabilities. Vocabulary softmax gives

\[
P(A)=P(B)=\frac{e}{2e+e^2}=\frac1{2+e}\approx0.21194,
\qquad
P(C)=\frac{e^2}{2e+e^2}=\frac e{2+e}\approx0.57612.
\]

The probabilities sum to approximately \(1.00000\), so greedy decoding would choose \(C\). A sampler could choose another token; decoding is a policy applied to the distribution, not another model layer.

If the observed next token is \(C\), the one-example cross-entropy is

\[
\mathcal L=-\log P(C)=-\log\left(\frac e{2+e}\right)\approx0.55144.
\]

## What this leaves out

This example omits the output projection \(W_O\), residual paths, feed-forward network, normalization, multiple heads, and learned biases. It is an end-to-end mathematical slice, not a second implementation of `DemoModel`. The Go model follows the same broad order but has different fixed numbers, a ReLU FFN, sinusoidal positions, and no normalization.

## Code map and checks

| Example step | Go location |
|---|---|
| lookup plus position | `go/minillm/model.go`, `Forward`, lines 50–56 |
| Q/K/V projections | `model.go`, `Forward`, lines 58–63 |
| causal scores and weighted values | `go/minillm/attention.go`, `CausalAttention` |
| stable vocabulary softmax | `go/minillm/math.go`, `Softmax` |
| future-position invariant | `go/minillm/math_test.go`, `TestCausalAttentionDoesNotReadFutureValues` |
| future-token invariant | `go/minillm/model_test.go`, `TestForwardStatesDoNotReadFutureTokens` |

Self-check:

1. Why does the final attention row have three weights while the first has one?
2. Which number is a logit, and which numbers are probabilities?
3. If \(W_Q\) and \(W_K\) were nonzero, which equation would stop producing uniform allowed weights?

<small>Source note: the sequence and parameters above are newly constructed for this guide. The attention equations follow Vaswani et al. (2017), and the source paper's attention discussion appears in sections 4–6.</small>
