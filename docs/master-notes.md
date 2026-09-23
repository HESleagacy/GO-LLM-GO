# Master Notes

This is the main route through the source paper. The paper is an explanatory overview, not a report of a new model or experiment. The notes preserve its teaching sequence while checking the math and separating intuition from formal guarantees.

## Argument in one page

An autoregressive language model assigns a probability to the next token given a prefix:

\[
P(w_{t+1}\mid w_1,\ldots,w_t).
\]

The model first maps each token to a vector. Each layer then computes context-sensitive vectors by projecting tokens into several learned spaces, comparing positions, normalizing the comparisons into weights, and taking weighted sums. A decoder applies a causal mask so position \(t\) cannot use tokens after \(t\). Repeated layers and nonlinear feed-forward transformations refine the representations. A final linear map produces one score per vocabulary item; softmax turns those scores into a distribution. Training changes the parameters to assign higher probability to observed next tokens.

## Navigation by source section

| Paper sections | Notes page | Main question |
|---|---|---|
| 1–3 | [Text, probability, and embeddings](notes/01-foundations.md) | What is predicted, and how do discrete tokens become vectors? |
| 4–5 | [Context and weighted averaging](notes/02-context-attention.md) | How can a token representation depend on other positions? |
| 6–7 | [Multiple relations and depth](notes/03-relations-depth.md) | Why use parallel attention subspaces and stacked layers? |
| 8–9 | [Position and next-token probabilities](notes/04-position-output.md) | How does order enter, and how do scores become probabilities? |
| 10 | [Training the parameters](notes/05-training.md) | What objective adjusts the model's parameters? |
| 11–13 | [Interpretation and claims](notes/06-interpretation.md), [equations](notes/07-equations.md) | What can be inferred from the mechanism, and what cannot? |

## Notation used here

- \(V\): vocabulary; \(|V|\): its number of token IDs.
- \(d\): model/embedding width; \(d_k\): query/key width; \(d_v\): value width.
- \(x_i\in\mathbb R^d\): representation at position \(i\).
- \(W_Q,W_K,W_V\): query, key, and value projections. The source paper calls these \(W^A,W^B,W^C\), describing their roles as receptive features, influence features, and content features.
- \(W_O\): output projection after concatenating parallel attention heads.
- \(L\): number of layers; \(H\): number of parallel heads.

The conventional Q/K/V names are included because they are used throughout software and research. The source's less metaphorical names remain useful for understanding the computation. Neither naming scheme changes the equations.

## Go implementation route

The [Go Lab](go-lab.md) implements one causal self-attention head plus the surrounding operations needed to produce vocabulary probabilities. Read each code function next to the equation it implements. The code is intentionally small enough to audit line by line.

## Reading rule

When a statement is an intuition, it is presented as intuition. When a statement is a mathematical consequence, its assumptions are stated. The [source notes](source-notes.md) call out paper passages that are useful pedagogically but too broad if read as universal technical facts.
