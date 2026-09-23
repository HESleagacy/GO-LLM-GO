# 4. Position, logits, and decoding

**Question:** How can the same token vectors represent different orderings, and how does a final hidden state become a next-token choice?

This follows Breeden §§8–9 and adds the implementation distinction between absolute sinusoidal positions and RoPE.

## Why position is necessary

Self-attention without a position signal treats the input as an ordered list only through the states it receives. If the same states are permuted, the corresponding outputs are permuted: the operation is permutation-equivariant. It has no built-in fact that “dog bites man” and “man bites dog” have different order. Position information must enter the states or the attention scores.

## Three position mechanisms

### Absolute addition

The simplest method adds a position vector:

\[
x_i^{(0)}=E[w_i]+p_i,
\qquad E[w_i],p_i\in\mathbb R^d.
\]

The Go demo uses a deterministic sinusoidal version:

\[
p_{i,2r}=\sin\left(i/10000^{2r/d}\right),\qquad
p_{i,2r+1}=\cos\left(i/10000^{2r/d}\right).
\]

This adds position before Q/K/V projections. It is easy to inspect and does not require a learned position table.

### RoPE

Rotary Position Embedding rotates each pair of query/key coordinates by a position-dependent angle. For a two-coordinate pair, rotation by \(\theta_i\) is

\[
R(\theta_i)=\begin{bmatrix}\cos\theta_i&-\sin\theta_i\\\sin\theta_i&\cos\theta_i\end{bmatrix},
\qquad q_i'=R(\theta_i)q_i,
\quad k_j'=R(\theta_j)k_j.
\]

The resulting dot product depends on the relative angle (and therefore relative positions) as well as content. RoPE is not “adding another embedding row,” and the Go demo does not implement it. See Su et al. (2021) for the primary description.

### Score bias

ALiBi adds a distance-dependent bias to attention scores instead of rotating or adding state vectors. It is another distinct design, not a correction to sinusoidal positions. The position method is part of the model specification: inference must use the same method used during training.

## Logits are scores, not choices

Let the final state at the last available position be \(h_t\in\mathbb R^d\). A vocabulary matrix \(U\in\mathbb R^{|V|\times d}\) and bias \(b\in\mathbb R^{|V|}\) produce

\[
\ell=Uh_t+b\in\mathbb R^{|V|},
\qquad
\ell_v=u_v^\top h_t+b_v.
\]

Each logit is an unrestricted real score. Softmax defines the model distribution:

\[
P_\theta(w_{t+1}=v\mid w_{1:t})
=\frac{\exp(\ell_v)}{\sum_{r\in V}\exp(\ell_r)}.
\]

For stability, compute the equivalent expression with \(m=\max_r\ell_r\):

\[
P(v)=\frac{\exp(\ell_v-m)}{\sum_{r\in V}\exp(\ell_r-m)}.
\]

Adding the same constant to all logits changes neither probability nor ranking. It is therefore harmless for stable softmax to subtract the maximum.

## Prediction versus decoding

The probability distribution is the model output. **Greedy decoding** chooses \(\arg\max_v P(v)\). **Sampling** draws a random token from the distribution. Temperature rescales logits before softmax, while top-k or nucleus sampling restricts the candidate set. These controls change the selected continuation, not the learned conditional distribution represented by the unmodified logits.

The [worked example](../worked-example.md) computes \(\ell=(1,1,2)\), probabilities approximately \((0.21194,0.21194,0.57612)\), and shows why greedy decoding chooses the third token without confusing it with softmax.

## Go connection

`Forward` adds `sinusoidalPosition`, uses only the final state for `MatVec(m.Output, ...)`, and calls `Softmax`. The output matrix is `[vocabulary, width]`; its rows are toy classifier vectors. `cmd/demo/main.go` only prints the distribution and its sum. It does not sample, append a token, or claim to generate text.

## Self-check

1. Why do two equal logits receive equal probabilities even if the other logits are different?
2. Is temperature part of the forward-pass probability model or a decoding control?
3. What must be changed if a checkpoint trained with RoPE is loaded into a sinusoidal-position implementation?

<small>Source: Breeden §§8–9. Primary references: Su et al., [*RoFormer*](https://arxiv.org/abs/2104.09864); Press et al., [*Train Short, Test Long*](https://arxiv.org/abs/2108.12409).</small>
