# PyTorch side-by-side lab

This chapter is a build plan for a separate repository that you create yourself. Keep it separate from these notes for three reasons:

1. The documentation repository stays dependency-free and remains a transparent Go reference.
2. PyTorch adds automatic differentiation, batching, GPU support, and checkpoint files that deserve their own tests and environment.
3. Comparing the same tiny example in Go and PyTorch gives you a reference implementation and catches shape or masking mistakes early.

The snippets below intentionally contain no code comments. Keep explanations in documentation and commit messages, then make the code express one operation clearly.

## Create the repository

Create a new repository, then run:

```sh
mkdir llm-learning-lab
cd llm-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install torch pytest
mkdir -p src/llm_lab tests notebooks configs experiments
touch src/llm_lab/__init__.py
```

Use this initial layout:

```text
llm-learning-lab/
├── src/llm_lab/
│   ├── __init__.py
│   ├── attention.py
│   ├── model.py
│   ├── training.py
│   ├── generation.py
│   └── checkpoint.py
├── tests/
├── notebooks/
├── configs/
└── experiments/
```

Use notebooks for inspection and plots. Put reusable operations in `src/llm_lab` so tests and experiments call the same code. Do not begin with `torch.nn.Transformer`; implement the small operations first so you can connect every tensor to an equation.

## Milestone 1: causal attention

**Reason:** attention is the central operation and the causal invariant is the most important correctness boundary. The Go function materializes only allowed positions; the PyTorch version can materialize a full score matrix and then mask it.

Create `src/llm_lab/attention.py`:

```python
import math

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    key_width = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(key_width)
    length = q.size(-2)
    blocked = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    scores = scores.masked_fill(blocked, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
```

Expected shapes are `q: [batch, sequence, key_width]`, `k: [batch, sequence, key_width]`, and `v: [batch, sequence, value_width]`. The output has shape `[batch, sequence, value_width]`; weights have shape `[batch, sequence, sequence]`.

Create `tests/test_attention.py`:

```python
import torch

from llm_lab.attention import causal_attention


def test_future_values_do_not_change_earlier_outputs():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    k = q.clone()
    v1 = torch.tensor([[[2.0, 0.0], [0.0, 4.0], [10.0, 10.0]]])
    v2 = torch.tensor([[[2.0, 0.0], [0.0, 4.0], [-1000.0, 1000.0]]])
    first, _ = causal_attention(q, k, v1)
    second, _ = causal_attention(q, k, v2)
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-12)
    assert not torch.allclose(first[:, 2], second[:, 2], atol=1e-12)


def test_attention_weights_sum_to_one():
    q = torch.randn(2, 4, 8)
    output, weights = causal_attention(q, q, q)
    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
```

Run:

```sh
PYTHONPATH=src pytest -q
```

Inspect `weights[0]` in a notebook. Every entry above the diagonal must be zero. This is the PyTorch equivalent of the Go causal-attention test.

## Milestone 2: embeddings and positions

**Reason:** separate token identity from order before adding attention. Start with the same sinusoidal absolute positions used by the Go demo, then implement learned positions and RoPE as separate experiments.

Create `src/llm_lab/model.py` with the position helper:

```python
import torch


def sinusoidal_positions(length: int, width: int, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    indices = torch.arange(width, device=device)
    exponent = (2 * torch.div(indices, 2, rounding_mode="floor")) / width
    angles = positions / (10000.0 ** exponent)
    result = torch.empty(length, width, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(angles[:, 0::2])
    result[:, 1::2] = torch.cos(angles[:, 1::2])
    return result
```

The position tensor has shape `[sequence, width]` and broadcasts across the batch. Verify position 0 and a later position by hand. Do not call this PCA: it is a deterministic coordinate construction, while token embeddings are learned parameters.

## Milestone 3: one decoder-shaped model

**Reason:** combine the tested pieces while keeping the architecture close to `DemoModel`. Start without normalization and with one head so the output can be compared to the Go path.

Add this model to `src/llm_lab/model.py`:

```python
from torch import nn
from torch.nn import functional as F

from .attention import causal_attention


class TinyDecoder(nn.Module):
    def __init__(self, vocabulary, width, hidden):
        super().__init__()
        self.vocabulary = vocabulary
        self.width = width
        self.embedding = nn.Embedding(vocabulary, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.feed_forward_in = nn.Linear(width, hidden, bias=False)
        self.feed_forward_out = nn.Linear(hidden, width, bias=False)
        self.output = nn.Linear(width, vocabulary, bias=False)

    def forward(self, token_ids):
        length = token_ids.size(-1)
        states = self.embedding(token_ids)
        states = states + sinusoidal_positions(
            length, self.width, states.device, states.dtype
        )
        q = self.query(states)
        k = self.key(states)
        v = self.value(states)
        context, weights = causal_attention(q, k, v)
        states = states + self.attention_output(context)
        states = states + self.feed_forward_out(F.relu(self.feed_forward_in(states)))
        return self.output(states), weights
```

The logits have shape `[batch, sequence, vocabulary]`. Unlike the Go demo, this uses learned embedding rows and randomly initialized PyTorch weights. For exact parity, copy the Go matrices into PyTorch tensors and assign them to the corresponding layers before comparing outputs.

## Milestone 4: next-token loss

**Reason:** training is a shifted-target problem, not a language-generation loop. Establish the batch contract before writing an optimizer.

Create `src/llm_lab/training.py`:

```python
import torch
from torch.nn import functional as F


def next_token_loss(logits: torch.Tensor, token_ids: torch.Tensor):
    predicted = logits[:, :-1].contiguous()
    targets = token_ids[:, 1:].contiguous()
    return F.cross_entropy(
        predicted.view(-1, predicted.size(-1)),
        targets.view(-1),
    )
```

For input `[w0, w1, w2]`, the model rows used for loss predict `[w1, w2]`. Test this with manually chosen logits where the target class is obvious. `CrossEntropyLoss` combines log-softmax and negative log-likelihood; do not apply softmax before passing logits to it.

## Milestone 5: train a toy corpus

**Reason:** a synthetic task separates implementation errors from data and modeling difficulty. Use a repeating sequence before natural language.

```python
import torch

from llm_lab.model import TinyDecoder
from llm_lab.training import next_token_loss


torch.manual_seed(7)
model = TinyDecoder(vocabulary=8, width=16, hidden=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7] * 8])

for step in range(300):
    logits, _ = model(tokens)
    loss = next_token_loss(logits, tokens)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Record the initial and final loss, but do not call the result a language-model benchmark. Add a validation sequence, save the seed and configuration, and verify that the model predicts the repeating pattern with greedy decoding.

## Milestone 6: parity before scale

Before adding multiple heads, LayerNorm, GELU, RoPE, or GPU code, compare the PyTorch model with the Go model on the same tiny matrices:

1. Copy embeddings and all Go matrices into PyTorch.
2. Use the same token IDs and sinusoidal positions.
3. Compare every intermediate state, attention output, residual output, logit, and probability.
4. Set a tolerance appropriate to the dtype and report the maximum absolute difference.
5. Change a future token and confirm earlier states and logits remain unchanged.

This order localizes errors. If you add all architecture features at once, a mismatch cannot tell you whether the problem is a transpose, mask, position formula, residual order, activation, or numerical dtype.

## Later milestones

After parity passes, implement one feature per branch or commit:

1. Add LayerNorm and choose pre-norm explicitly.
2. Replace ReLU with GELU and compare training behavior.
3. Add multiple heads, concatenation, and `W_O`.
4. Add a real tokenizer and licensed text corpus.
5. Add validation, checkpoints, and greedy generation.
6. Add sampling controls separately from model logits.
7. Add KV caching and compare cached versus uncached logits.
8. Run the multilingual language-effect benchmark.
9. Try fine-tuning only after a held-out evaluation exists.

At each milestone, retain a small test that can be checked by hand. The aim is to understand why the model works before increasing its size.
