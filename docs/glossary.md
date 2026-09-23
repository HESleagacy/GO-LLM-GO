# Glossary

**Autoregressive**: a model that predicts the next token from tokens already in the prefix.

**Causal mask**: a rule that blocks attention from query position \(i\) to future key positions \(j>i\). “Causal” here means no future information is available, not that attention is a scientific causal-effect estimate.

**Contextual representation**: a vector at a position after operations that can use surrounding tokens. It can differ for the same token ID in different sentences.

**Embedding**: a learned vector lookup for a discrete ID. It is a parameter table, not automatically PCA or a dictionary of human-readable features.

**Head**: one set of query, key, and value projections inside multi-head attention. The equations allow heads to learn different patterns; they do not guarantee a clean semantic job per head.

**Key**: the projected features a position exposes for query-key compatibility comparisons.

**KV cache**: stored key and value projections from earlier decoding steps. It avoids recomputing them; it does not remove the need to compute the new query or change the model distribution.

**Layer normalization (LayerNorm)**: a normalization operation that uses the mean and variance of features for a position, followed by learned scale and shift parameters. RMSNorm is a related variant that omits mean subtraction.

**Logit**: an unrestricted real-valued score before vocabulary softmax. It is not a probability.

**Position encoding**: information that lets the model distinguish order. The Go demo adds sinusoidal absolute vectors; RoPE instead rotates query and key pairs by position-dependent angles.

**Query**: the projected features used by a position to ask which visible keys match.

**Residual connection**: an addition such as \(x+f(x)\), giving a sublayer an identity route around its transformation.

**Softmax**: the map from scores to positive values summing to one, \(\operatorname{softmax}(z)_i=e^{z_i}/\sum_j e^{z_j}\).

**Token**: one vocabulary symbol produced by a tokenizer. It may be a word, subword, punctuation mark, whitespace marker, or special symbol.

**Value**: the projected content that is actually averaged after attention weights have been computed.

See [Notation and dimensions](notation.md) for shapes and [Source and scope](source-notes.md) for claim boundaries.
