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

## A parallel learning track

Use the roadmap below to turn each concept in these notes into a small experiment. Implement one narrow piece, write an invariant test, and compare its output with the equations before moving on. The goal is understanding and reproducibility, not building a production system immediately.

### 1. Learn a real tokenizer

Study byte-pair encoding (BPE), unigram tokenization, special tokens, vocabulary files, and encode/decode reversibility. Implement a deliberately small tokenizer first:

- normalize a documented input format;
- split training text into symbols or bytes;
- count pair frequencies and merge the most frequent pair;
- save the merge rules and vocabulary;
- encode text to IDs and decode IDs back to text.

Test whitespace, punctuation, repeated characters, unknown bytes, empty input, and special tokens. Measure token counts by language; tokenizer fragmentation is an important confounder in the language-effect benchmark.

### 2. Move the forward pass to PyTorch

Recreate the Go equations with tensors before attempting a large model. Use `nn.Embedding`, explicit Q/K/V linear layers, a lower-triangular causal mask, a stable vocabulary softmax or `CrossEntropyLoss`, residual connections, and a deliberately chosen pre-norm block. Print tensor shapes at every boundary.

Compare a fixed tiny example against this repository's hand calculation. Use `torch.float64` initially so numerical differences are easy to diagnose. Then compare a small `float32` implementation with a stated tolerance.

### 3. Learn automatic differentiation and optimizers

Start with scalar functions and finite-difference checks. Then verify gradients for a matrix-vector product, ReLU or GELU, softmax cross-entropy, and one attention row. Compare manual derivatives with PyTorch autograd.

Train a tiny character- or token-level model on a toy corpus. Log training and validation loss, use AdamW only after plain gradient descent is understood, and record learning rate, batch size, seed, parameter count, and checkpoint version. A falling training loss alone is not evidence of useful generalization.

### 4. Build the data pipeline

Learn dataset splits, minibatching, padding, attention masks, sequence packing, and next-token target shifting. Make the batch contract explicit:

\[
\text{input}[b,t]=w_t,\qquad \text{target}[b,t]=w_{t+1}.
\]

Test that padding contributes no loss, that target tokens are shifted exactly once, and that no batch contains future information through preprocessing. Record dataset provenance, license, preprocessing code, and a content hash.

### 5. Save, load, and verify checkpoints

Study state dictionaries, serialization formats, configuration files, checksums, and version compatibility. Save the tokenizer and model configuration with the weights. On load, reject missing keys, unexpected keys, wrong shapes, non-finite values, unsupported versions, and vocabulary mismatches.

Keep fixed input IDs and expected logits as a checkpoint parity test. Loading a checkpoint is correct only when the inference implementation reproduces the training implementation within an explicit tolerance.

## Honest status of this repository

The repository currently reaches only the **forward-pass demonstration** stage. It does not yet have a tokenizer, training, checkpoint loading, generation loop, or KV cache. Once those pieces exist and a checkpoint is trained, calling it a small language model is reasonable. Until then, “inference-ready” describes this roadmap, not the current code.
