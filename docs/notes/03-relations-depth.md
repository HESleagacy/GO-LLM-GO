# 3. Multiple relations and depth

This page follows source sections 6–7: run several attention computations in parallel, combine them, and stack layers with feed-forward transformations.

## 3.1 Parallel attention heads

One projection set can learn one compatibility pattern. A Transformer layer uses \(H\) separate sets of projections. For head \(h\):

\[
z_i^{(h)}=\sum_{j\in A_i}\alpha_{ij}^{(h)}W_V^{(h)}x_j,
\qquad
y_i=W_O[z_i^{(1)};\ldots;z_i^{(H)}].
\]

The semicolon denotes concatenation. The output projection maps the concatenated vector back to the model width. Each head has its own learned projections, so different patterns can be represented in parallel.

The source calls a head a “relation” to emphasize the computation rather than an assigned semantic role. That is a useful translation for intuition. The standard engineering term remains **attention head**.

!!! note "Do not assume clean specialization"
    Multiple heads create capacity for different patterns; training does not guarantee that each head learns a distinct linguistic relation. Heads may specialize, overlap, or be partly redundant. “One head = one grammatical job” is a story, not a property enforced by the equations.

## 3.2 Stacking layers expands the receptive computation

Let \(x_i^{(0)}\) be the input embedding plus position information. A layer takes the whole sequence of states and returns new states:

\[
x_i^{(\ell)}=\operatorname{Layer}_{\ell}(x_1^{(\ell-1)},\ldots,x_n^{(\ell-1)})_i.
\]

With causal attention, after one layer position \(i\) can use positions up to \(i\). After multiple layers it can combine features that earlier positions themselves gathered from their prefixes. The effective computation becomes richer without requiring a single operation to encode every relationship.

## 3.3 The feed-forward transformation

After attention, a position-wise nonlinear network transforms each vector independently:

\[
\operatorname{FFN}(x)=W_2\,\phi(W_1x+b_1)+b_2.
\]

The same FFN weights are applied at each position within a layer. The nonlinearity \(\phi\) matters: without a nonlinearity, a composition of linear maps remains linear and cannot create the same expressive features.

The paper illustrates ReLU, \(\max(0,x)\). Modern architectures also use alternatives such as GELU or gated activations. The exact choice is an implementation/model-design choice.

## 3.4 Residual paths and normalization

A residual connection adds a transformation to its input, schematically:

\[
x' = x + f(x).
\]

The identity path makes it easier for information and gradients to pass through deep stacks. This is a helpful route for gradient flow, not a guarantee that gradients can never vanish or that optimization is automatically stable.

Normalization keeps activation scales manageable. LayerNorm and RMSNorm are common variants; architectures differ in whether normalization is before or after a sublayer. The source's “mean 0 and variance 1 after each sub-operation” describes one simplified picture, not a universal Transformer recipe.

## 3.5 Parameter sharing boundaries

The attention heads and layers have learned parameters. Typically, the token embedding table may also be tied to the output projection, but tying is optional and the paper lists separate input and output vectors. Do not infer a specific implementation choice from the high-level diagram alone.

## Go connection

The included program focuses on one head so the dot products and weighted sum stay inspectable. The same function can be repeated with separate projection matrices and concatenated outputs to build multiple heads. A fully configurable multi-layer trainer would add substantial machinery that the paper explains conceptually but the toy forward pass does not implement.
