# Master Notes

## The argument in one page

The chain rule factors a sequence probability:

\[
P(w_{1:n})=\prod_{t=1}^{n}P(w_t\mid w_{<t}).
\]

A decoder-only model learns one reusable parameterized function for these conditional distributions. It looks up each discrete token in an embedding table, adds position information, and repeatedly transforms the resulting states. In self-attention, query-key compatibility determines which visible positions matter; values carry the content that is averaged. A causal mask enforces \(j\le i\). Feed-forward networks transform each position independently, while residual paths and normalization help organize a deep stack. A vocabulary projection produces logits, and softmax turns those scores into probabilities.

The equations explain the computation. They do not, by themselves, prove that a head has a named linguistic role, that a high-probability statement is true, or that a trained model performs formal reasoning.

## Navigation by question

| Question | Page | Source / added material |
|---|---|---|
| What is predicted, and how do IDs become vectors? | [Foundations](notes/01-foundations.md) | Breeden §§1–3; embedding clarification |
| How does one position use context? | [Attention](notes/02-context-attention.md) | Breeden §§4–5; Vaswani et al. |
| Why heads, depth, residuals, and norms? | [Depth](notes/03-relations-depth.md) | Breeden §§6–7; architecture variants |
| How does order enter and how are scores decoded? | [Position and output](notes/04-position-output.md) | Breeden §§8–9; RoPE reference |
| How are parameters fitted and used? | [Training](notes/05-training.md) | Breeden §10; cache clarification |
| What claims are safe? | [Interpretation](notes/06-interpretation.md) | Breeden §§11–13; claim labels |
| Can I check the algebra? | [Worked example](worked-example.md) | New, reproducible calculation |

## Notation contract

The model width is \(d\), key/query width is \(d_k\), value width is \(d_v\), sequence length is \(n\), vocabulary size is \(|V|\), layers are \(L\), and heads are \(H\). See [Notation and dimensions](notation.md) for every matrix shape. The paper uses 1-based indices; Go uses 0-based indices.

The paper denotes the three attention maps as \(W^A,W^B,W^C\); these notes use the conventional \(W_Q,W_K,W_V\). They are the same roles under different names.

## One important boundary

The repository's Go model is a transparent forward pass, not a small trained language model. It has no tokenizer, corpus, automatic differentiation, optimizer, checkpoint, generation loop, or KV cache. A normalized output vector proves only that the implemented softmax is a distribution; it says nothing about language quality.
