# Inference readiness: from toy forward pass to a usable small model

“Inference-ready” means more than returning a normalized vector. A usable language model must load trained parameters, map real text to token IDs, run the same architecture those parameters were trained for, and decode output safely and predictably. The current Go demo has the shape of part of that path, but it is not trained and cannot generate meaningful language.

## Required before useful text generation

### 1. Fix the model specification

Define and persist the exact architecture and vocabulary contract:

- vocabulary size, token IDs, special-token IDs, model width, layer count, head count, context limit, and feed-forward width;
- positional method and its parameters (the current demo uses sinusoidal absolute positions; a checkpoint trained with RoPE requires RoPE at inference);
- normalization type and placement, activation, projection biases, and whether input embeddings share weights with the output projection;
- numeric format and any quantization scheme.

The inference implementation must match training. A checkpoint with different dimensions or positional behavior is not “close enough”; it computes a different function.

### 2. Add a tokenizer and text boundary

Replace the five placeholder IDs with a tokenizer that has a stable vocabulary and exact encode/decode behavior. It must handle unknown or byte-level input according to its design, preserve special-token semantics, and share the same vocabulary file used during training. Add round-trip tests for representative text, punctuation, whitespace, and non-ASCII input.

### 3. Train parameters and save a checkpoint

Add an actual training path: corpus preparation, next-token examples, causal masking, cross-entropy loss, backpropagation/autodiff, optimizer updates, validation, and checkpoint saving. Save all learned tensors and the model/tokenizer configuration together. Record the training-data source and license. A random or hand-authored parameter set is not a substitute for a trained checkpoint.

### 4. Load and validate the checkpoint

Implement a loader that rejects missing tensors, unexpected shapes, unsupported formats, and non-finite values. Add a checkpoint version and checksum. Verify the loaded model against a small set of fixed input IDs and expected logits/probabilities produced by the training implementation. This catches architecture drift between training and inference.

### 5. Implement autoregressive decoding

Generation repeats the following loop:

1. Encode the prompt into token IDs.
2. Run the model on the available context and obtain the final-position logits.
3. Apply decoding controls such as temperature and optional top-k or nucleus sampling.
4. Choose the next token, stop on an end-of-sequence token or length limit, append the token, and continue.
5. Decode generated IDs back into text.

Keep greedy decoding available for deterministic tests. Sampling options change which continuation is selected; they do not change the model's learned probability distribution.

### 6. Enforce context and numerical rules

- Reject or explicitly truncate prompts beyond the trained context limit; never silently index outside positional state.
- Apply the causal mask at every layer.
- Use stable softmax/log-sum-exp and reject NaN or infinite parameters and logits.
- Define behavior for empty prompts, end-of-sequence, maximum generation length, and all-masked rows.
- Test that appending a future token cannot alter logits for earlier positions under causal evaluation.

## Needed for practical repeated inference

### KV cache

The current demo recomputes all previous positions each time. A key/value cache stores the keys and values from earlier tokens so each generation step processes only the new token. This is a performance improvement, not a correctness requirement. First verify cached and uncached decoding return matching logits within a documented numeric tolerance.

### Resource limits and runtime behavior

Measure the actual checkpoint on the intended CPU or GPU instance. Record peak memory, model-load time, prompt throughput, token-generation latency, and maximum supported context. Add batch-size and request-length limits so an instance cannot be exhausted by an oversized request. For a small CPU model, a GPU may cost more than it saves; benchmark rather than guessing.

### Evaluation and release checks

- Compare logits with a trusted reference implementation on fixed prompts.
- Measure held-out negative log-likelihood/perplexity; a pretty sample alone is not evaluation.
- Test deterministic greedy outputs and seeded sampling behavior.
- Test tokenizer parity, checkpoint corruption, context boundaries, EOS handling, and cache parity.
- Record the model version, tokenizer version, configuration, and known limitations with every release.

## Suggested implementation order

| Stage | Deliverable | Exit check |
|---|---|---|
| A. Real input/output | Stable tokenizer and text encode/decode | Round-trip and special-token tests pass |
| B. Trainable model | Autodiff, loss, optimizer, small corpus | Training loss falls and held-out loss is finite |
| C. Checkpoint parity | Save/load configuration and weights | Inference logits match training logits |
| D. Text generation | Greedy decode, EOS, length/context rules | Repeatable output; edge-case tests pass |
| E. Efficient serving | KV cache and resource controls | Cached logits match uncached logits; instance benchmark recorded |

## Honest status of this repository

The repository currently reaches only the **forward-pass demonstration** stage. It does not yet have a tokenizer, training, checkpoint loading, generation loop, or KV cache. Once those pieces exist and a checkpoint is trained, calling it a small language model is reasonable. Until then, “inference-ready” describes this roadmap, not the current code.
