# PyTorch side-by-side lab

This chapter is a build plan for a separate repository that you create yourself. Keep it separate from these notes for three reasons:

1. The documentation repository stays dependency-free and remains a transparent Go reference.
2. PyTorch adds automatic differentiation, batching, GPU support, and checkpoint files that deserve their own tests and environment.
3. Comparing the same tiny example in Go and PyTorch gives you a reference implementation and catches shape or masking mistakes early.

The snippets below intentionally contain no code comments. Keep explanations in documentation and commit messages, then make the code express one operation clearly. Every file on this page is complete: copy it as shown, run the stated command, and compare against the stated result before moving on.

## Create the repository

Create a new repository, then run:

```sh
mkdir llm-learning-lab
cd llm-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install torch pytest
mkdir -p src/llm_lab tests/fixtures notebooks configs experiments
touch src/llm_lab/__init__.py
```

On a machine without a GPU, `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` installs a much smaller CPU-only build.

The finished layout after Milestone 11 is:

```text
llm-learning-lab/
├── src/llm_lab/
│   ├── __init__.py
│   ├── attention.py        Milestones 1, 9, 10
│   ├── model.py            Milestones 2–3, 9, 10
│   ├── training.py         Milestones 4–5
│   ├── go_parity.py        Milestone 6
│   ├── generation.py       Milestones 7, 10
│   ├── checkpoint.py       Milestone 8
│   └── tokenizer.py        Milestone 11
├── tests/
│   ├── fixtures/demo_reference.json
│   ├── test_pytorch_basics.py
│   ├── test_attention.py
│   ├── test_positions.py
│   ├── test_training.py
│   ├── test_gradients.py
│   ├── test_go_parity.py
│   ├── test_generation.py
│   ├── test_checkpoint.py
│   ├── test_multi_head.py
│   ├── test_kv_cache.py
│   └── test_tokenizer.py
├── experiments/
│   └── train_repeating.py
├── notebooks/
└── configs/
```

Use notebooks for inspection and plots. Put reusable operations in `src/llm_lab` so tests and experiments call the same code. Do not begin with `torch.nn.Transformer`; implement the small operations first so you can connect every tensor to an equation.

Run every test from the repository root with:

```sh
PYTHONPATH=src pytest -q
```

## Milestone 0: PyTorch essentials

**Reason:** every later milestone assumes these five ideas. If one of them is unfamiliar, the bugs it causes look like model bugs.

| Idea | What to know | Go equivalent |
|---|---|---|
| Tensor and shape | An n-dimensional array; `x.shape` is the first thing to print when anything goes wrong. `view`/`reshape` change shape without changing data. | `Matrix` / `Vector` |
| Broadcasting | Size-1 or missing leading dimensions are stretched automatically: `[2,3] + [3]` adds the row to every row. This is how one position table is added to a whole batch. | an explicit loop |
| `@` on batches | `[batch, n, k] @ [k, m] → [batch, n, m]`: the last two dimensions are multiplied, the rest are looped over. | `MatVec` in a loop |
| `nn.Linear(in, out)` | Holds `weight` of shape `[out, in]` and computes \(Wx\). | `MatVec(W, x)` |
| `nn.Module` | A class whose `nn.Linear`/`nn.Embedding` attributes are registered automatically, so `model.parameters()` finds every weight for the optimizer. | the `Model` struct |
| Autograd | A tensor with `requires_grad=True` records operations; `loss.backward()` fills `.grad` on every such tensor with \(\partial\,\text{loss}/\partial\,\text{tensor}\). Inside `torch.no_grad()` nothing is recorded. | none; Go has no gradients |
| `dtype` | `float32` is the default; use `float64` when comparing against hand calculations or Go. | always `float64` |

Create `tests/test_pytorch_basics.py` and make it pass before Milestone 1. Each test pins down one row of the table with a number you can check by hand. In the autograd test, \(\frac{d}{dw}(2w-1)^2 = 4(2w-1) = 20\) at \(w=3\).

```python
import torch
from torch import nn


def test_shapes_and_broadcasting():
    x = torch.arange(6.0).view(2, 3)
    assert x.shape == (2, 3)
    assert x[1].tolist() == [3.0, 4.0, 5.0]
    row = torch.tensor([10.0, 20.0, 30.0])
    assert (x + row).tolist() == [[10.0, 21.0, 32.0], [13.0, 24.0, 35.0]]
    assert (x @ x.T).shape == (2, 2)
    batch = torch.randn(4, 5, 3)
    assert (batch @ torch.randn(3, 7)).shape == (4, 5, 7)


def test_linear_stores_out_by_in():
    layer = nn.Linear(3, 2, bias=False)
    assert layer.weight.shape == (2, 3)
    x = torch.randn(3)
    assert torch.allclose(layer(x), layer.weight @ x)


def test_autograd_computes_derivatives():
    w = torch.tensor(3.0, requires_grad=True)
    loss = (2 * w - 1) ** 2
    loss.backward()
    assert w.grad.item() == 2 * 2 * (2 * 3.0 - 1)
    with torch.no_grad():
        frozen = w * 2
    assert not frozen.requires_grad


def test_module_registers_parameters():
    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 8)
            self.second = nn.Linear(8, 2)

        def forward(self, x):
            return self.second(torch.relu(self.first(x)))

    model = Pair()
    assert sum(p.numel() for p in model.parameters()) == (4 * 8 + 8) + (8 * 2 + 2)
    assert model(torch.randn(5, 4)).shape == (5, 2)
```

**Exit check:** 4 tests pass and you can predict the shape of every result before running it.

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

Expected shapes are `q: [batch, sequence, key_width]`, `k: [batch, sequence, key_width]`, and `v: [batch, sequence, value_width]`. The output has shape `[batch, sequence, value_width]`; weights have shape `[batch, sequence, sequence]`. The diagonal is never masked, so every row has at least one finite score and softmax never sees an all-`-inf` row.

Create `tests/test_attention.py`. The third test reproduces the [worked example](worked-example.md) exactly: zero queries and keys give uniform weights over the allowed positions, and the output rows are \((1,0)\), \((1,\tfrac12)\), \((1,1)\).

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


def test_worked_example_uniform_weights():
    x = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]], dtype=torch.float64)
    zeros = torch.zeros_like(x)
    output, weights = causal_attention(zeros, zeros, x)
    expected_weights = torch.tensor(
        [[1, 0, 0], [1 / 2, 1 / 2, 0], [1 / 3, 1 / 3, 1 / 3]], dtype=torch.float64
    )
    expected_output = torch.tensor([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(weights[0], expected_weights)
    assert torch.allclose(output[0], expected_output)
```

**Exit check:** 3 tests pass. Inspect `weights[0]` in a notebook; every entry above the diagonal must be exactly zero. This is the PyTorch equivalent of `TestCausalAttentionDoesNotReadFutureValues` in Go.

## Milestone 2: embeddings and positions

**Reason:** separate token identity from order before adding attention. Start with the same sinusoidal absolute positions used by the Go demo, then implement learned positions and RoPE as separate experiments.

The position helper goes at the top of `src/llm_lab/model.py`; the complete file is shown in Milestone 3. It computes, for position \(t\) and coordinate \(i\),

\[
p_{t,i}=\begin{cases}\sin\!\left(t/10000^{2\lfloor i/2\rfloor/d}\right) & i\text{ even}\\ \cos\!\left(t/10000^{2\lfloor i/2\rfloor/d}\right) & i\text{ odd},\end{cases}
\]

which is the same formula as `sinusoidalPosition` in `go/minillm/math.go`.

Create `tests/test_positions.py`. For width 4, position 0 is \((0,1,0,1)\) and position 1 is \((\sin 1,\cos 1,\sin 0.01,\cos 0.01)\), because the second pair uses \(10000^{2/4}=100\).

```python
import math

import torch

from llm_lab.model import sinusoidal_positions


def test_hand_checked_positions():
    table = sinusoidal_positions(2, 4, "cpu", torch.float64)
    assert table.shape == (2, 4)
    assert torch.allclose(table[0], torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64))
    expected = torch.tensor(
        [math.sin(1.0), math.cos(1.0), math.sin(0.01), math.cos(0.01)],
        dtype=torch.float64,
    )
    assert torch.allclose(table[1], expected)
```

The position tensor has shape `[sequence, width]` and broadcasts across the batch. Do not call this PCA: it is a deterministic coordinate construction, while token embeddings are learned parameters.

## Milestone 3: one decoder-shaped model

**Reason:** combine the tested pieces while keeping the architecture close to `DemoModel`. Start without normalization and with one head so the output can be compared to the Go path.

Write the complete `src/llm_lab/model.py`:

```python
import torch
from torch import nn
from torch.nn import functional as F

from .attention import causal_attention


def sinusoidal_positions(length: int, width: int, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    indices = torch.arange(width, device=device)
    exponent = (2 * torch.div(indices, 2, rounding_mode="floor")) / width
    angles = positions / (10000.0 ** exponent)
    result = torch.empty(length, width, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(angles[:, 0::2])
    result[:, 1::2] = torch.cos(angles[:, 1::2])
    return result


class TinyDecoder(nn.Module):
    def __init__(self, vocabulary, width, hidden):
        super().__init__()
        self.vocabulary = vocabulary
        self.width = width
        self.hidden = hidden
        self.embedding = nn.Embedding(vocabulary, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.feed_forward_in = nn.Linear(width, hidden, bias=False)
        self.feed_forward_out = nn.Linear(hidden, width, bias=False)
        self.output = nn.Linear(width, vocabulary, bias=False)

    def config(self):
        return {"vocabulary": self.vocabulary, "width": self.width, "hidden": self.hidden}

    def states(self, token_ids):
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
        return states, weights

    def forward(self, token_ids):
        states, weights = self.states(token_ids)
        return self.output(states), weights
```

`states` returns the per-position states before the vocabulary projection, mirroring `ForwardStates` in Go; `forward` adds the vocabulary logits with shape `[batch, sequence, vocabulary]`. `config` records the three numbers needed to rebuild the model when loading a checkpoint.

Each layer maps one-to-one onto a Go matrix, and the storage convention already agrees. `nn.Linear(in, out).weight` has shape `[out, in]` and computes \(Wx\), exactly like a row-major Go `Matrix` passed to `MatVec`. No transpose is needed when copying weights:

| Go field | Go shape | PyTorch attribute | `weight` shape |
|---|---|---|---|
| `Embeddings` | `[vocabulary, width]` | `embedding` | `[vocabulary, width]` |
| `WQ`, `WK`, `WV` | `[width, width]` | `query`, `key`, `value` | `[width, width]` |
| `WO` | `[width, width]` | `attention_output` | `[width, width]` |
| `W1` | `[hidden, width]` | `feed_forward_in` | `[hidden, width]` |
| `W2` | `[width, hidden]` | `feed_forward_out` | `[width, hidden]` |
| `Output` | `[vocabulary, width]` | `output` | `[vocabulary, width]` |

Unlike the Go demo, a freshly constructed `TinyDecoder` has randomly initialized weights. Milestone 6 replaces them with the Go values.

## Milestone 4: next-token loss

**Reason:** training is a shifted-target problem, not a language-generation loop. Establish the batch contract before writing an optimizer.

Create `src/llm_lab/training.py`. `evaluate_loss` is used from Milestone 5 onwards to measure loss without building a gradient graph.

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


@torch.no_grad()
def evaluate_loss(model, token_ids):
    model.eval()
    logits, _ = model(token_ids)
    model.train()
    return next_token_loss(logits, token_ids).item()
```

For input `[w0, w1, w2]`, the model rows used for loss predict `[w1, w2]`; the final row has no target and is dropped. `CrossEntropyLoss` combines log-softmax and negative log-likelihood, so pass logits, not probabilities.

Create `tests/test_training.py` with two known answers. In the first, rows 0 and 1 put almost all mass on the shifted targets 1 and 2, while row 2 is deliberately wrong; the loss is still near zero, which proves the last row is ignored. In the second, uniform logits over 5 classes give \(-\log(1/5)=\log 5\approx1.6094\) for every row.

```python
import math

import torch

from llm_lab.training import next_token_loss


def test_targets_are_shifted_once():
    token_ids = torch.tensor([[0, 1, 2]])
    logits = torch.full((1, 3, 3), -20.0)
    logits[0, 0, 1] = 20.0
    logits[0, 1, 2] = 20.0
    logits[0, 2, 0] = -100.0
    assert next_token_loss(logits, token_ids).item() < 1e-6


def test_uniform_logits_give_log_vocabulary():
    token_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros(1, 4, 5)
    assert math.isclose(next_token_loss(logits, token_ids).item(), math.log(5), rel_tol=1e-6)
```

**Exit check:** both tests pass. If the first one fails, the shift is off by one or the last row is being scored.

### Why training works

Training repeats four lines. Each one has a precise meaning:

| Line | What happens |
|---|---|
| `logits, _ = model(tokens)` then `loss = next_token_loss(...)` | Forward pass. Autograd records every operation from the parameters to one scalar loss. |
| `optimizer.zero_grad(set_to_none=True)` | Clears the previous step's gradients. `.grad` *accumulates* across `backward()` calls, so forgetting this adds stale gradients. |
| `loss.backward()` | Applies the chain rule backwards through the recorded operations and stores \(\partial\mathcal L/\partial\theta\) in every parameter's `.grad`. |
| `optimizer.step()` | Moves each parameter a small amount against its gradient. |

**The gradient at the output is simple.** For logits \(\ell\), probabilities \(p=\operatorname{softmax}(\ell)\), and correct class \(y\),

\[
\frac{\partial\mathcal L}{\partial\ell_v}=p_v-\mathbf 1[v=y].
\]

The correct logit is pushed up by \(1-p_y\); every wrong logit is pushed down by its own probability. A confident correct prediction produces almost no gradient, which is why loss falls quickly at first and then slowly. Backpropagation passes this signal through `output`, the residual paths, the feed-forward layers, attention, and finally the embedding rows of the tokens that appeared.

**Gradient descent** with learning rate \(\eta\) is

\[
\theta\leftarrow\theta-\eta\,\frac{\partial\mathcal L}{\partial\theta}.
\]

For a small enough \(\eta\), this lowers the loss because the gradient is the direction of steepest local increase. Too large an \(\eta\) overshoots and the loss rises or becomes NaN.

**AdamW**, used in Milestone 5, keeps a running mean \(m\) and running mean square \(s\) of each parameter's gradient and steps by \(\eta\,m/(\sqrt{s}+\epsilon)\). That gives every parameter a step of roughly size \(\eta\) regardless of its gradient's scale. The "W" is decoupled weight decay: a separate small shrink \(\theta\leftarrow\theta-\eta\lambda\theta\) that is not mixed into the gradient.

Create `tests/test_gradients.py`. It checks the output-gradient formula above, compares autograd against a finite-difference estimate \(\frac{\mathcal L(w+h)-\mathcal L(w-h)}{2h}\) for every weight of a small layer, and confirms that one manual gradient-descent step lowers the loss:

```python
import torch
from torch.nn import functional as F


def test_cross_entropy_gradient_is_probabilities_minus_target():
    logits = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
    target = torch.tensor([0])
    F.cross_entropy(logits, target).backward()
    expected = torch.softmax(logits.detach(), dim=-1) - F.one_hot(target, 3)
    assert torch.allclose(logits.grad, expected)


def test_autograd_matches_finite_differences():
    torch.manual_seed(0)
    x = torch.randn(4, dtype=torch.float64)
    w = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1])

    def loss_of(weights):
        return F.cross_entropy((weights @ x).unsqueeze(0), target)

    loss_of(w).backward()
    step = 1e-6
    numeric = torch.zeros_like(w)
    with torch.no_grad():
        for i in range(3):
            for j in range(4):
                plus, minus = w.clone(), w.clone()
                plus[i, j] += step
                minus[i, j] -= step
                numeric[i, j] = (loss_of(plus) - loss_of(minus)) / (2 * step)
    assert torch.allclose(w.grad, numeric, atol=1e-8)


def test_one_gradient_step_lowers_the_loss():
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    target = torch.randint(0, 3, (8,))
    w = torch.zeros(3, 4, requires_grad=True)
    before = F.cross_entropy(x @ w.T, target)
    before.backward()
    with torch.no_grad():
        w -= 0.5 * w.grad
    after = F.cross_entropy(x @ w.T, target)
    assert after < before
```

**Exit check:** 3 tests pass. The finite-difference test is the general tool: whenever you write a custom operation, check its gradient this way before trusting it.

## Milestone 5: train a toy corpus

**Reason:** a synthetic task separates implementation errors from data and modeling difficulty. Use a repeating sequence before natural language.

Create `experiments/train_repeating.py`. It trains on the cycle `0 1 2 … 7` repeated 8 times and validates on the same cycle starting at a different phase (`3 4 5 6 7 0 1 2 …`), so every validation token appears at a different absolute position than during training. It uses `greedy_generate` and `save_checkpoint` from Milestones 7 and 8, so write those two files before running it.

```python
import json
from pathlib import Path

import torch

from llm_lab.checkpoint import save_checkpoint
from llm_lab.generation import greedy_generate
from llm_lab.model import TinyDecoder
from llm_lab.training import evaluate_loss, next_token_loss

CONFIG = {"seed": 7, "vocabulary": 8, "width": 16, "hidden": 32, "lr": 3e-3, "steps": 300}
CONTEXT_LIMIT = 64

torch.manual_seed(CONFIG["seed"])
model = TinyDecoder(CONFIG["vocabulary"], CONFIG["width"], CONFIG["hidden"])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
cycle = list(range(CONFIG["vocabulary"]))
train_tokens = torch.tensor([cycle * 8])
validation_tokens = torch.tensor([(cycle[3:] + cycle[:3]) * 4])

history = {"initial_train": evaluate_loss(model, train_tokens),
           "initial_validation": evaluate_loss(model, validation_tokens)}
for step in range(CONFIG["steps"]):
    logits, _ = model(train_tokens)
    loss = next_token_loss(logits, train_tokens)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
history["final_train"] = evaluate_loss(model, train_tokens)
history["final_validation"] = evaluate_loss(model, validation_tokens)

generated = greedy_generate(model, torch.tensor([[5, 6]]), 10, CONTEXT_LIMIT)
history["greedy_from_5_6"] = generated[0].tolist()

output = Path("experiments/runs/repeating")
output.mkdir(parents=True, exist_ok=True)
(output / "result.json").write_text(json.dumps({"config": CONFIG, **history}, indent=2))
save_checkpoint(output / "model.pt", model, CONFIG)
print(json.dumps(history, indent=2))
```

Run:

```sh
PYTHONPATH=src python experiments/train_repeating.py
```

With PyTorch 2.14 on CPU this printed:

```text
initial_train       2.454
initial_validation  2.513
final_train         0.0004
final_validation    0.043
greedy_from_5_6     [5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0]
```

The initial losses should be close to \(\log 8\approx2.08\) (uniform guessing) plus some noise from random weights. Exact values will differ across PyTorch versions and hardware, but you should see the same pattern: training loss falls by several orders of magnitude, validation loss stays small but higher than training loss, and greedy decoding continues the cycle. If validation loss stays near 2, the model has memorized positions rather than learned the token transition.

`experiments/runs/repeating/` now holds `result.json` (configuration and losses), `model.pt`, and `model.sha256`. Do not call this a language-model benchmark: the task can be solved from the current token alone.

## Milestone 6: parity with the Go model

**Reason:** an independent implementation is the cheapest way to find a transpose, mask, position, or residual-order bug. Do this before adding any new architecture feature.

### Export the Go reference

In this documentation repository, `go/cmd/export` writes `DemoModel`'s parameters together with its states, final logits, and probabilities for the token IDs `[0, 2, 1]`:

```sh
cd go
go run ./cmd/export > ../../llm-learning-lab/tests/fixtures/demo_reference.json
```

Adjust the output path to wherever your lab repository lives. The JSON has this shape:

```text
{
  "parameters": {"Embeddings": [[...]], "WQ": [[...]], ..., "Output": [[...]]},
  "token_ids": [0, 2, 1],
  "states": [[...], [...], [...]],
  "final_logits": [1.28336714328134, 0.505198846501708, 0.5599066316258898, 1.246051035221093, -0.25636815416117326],
  "probabilities": [0.3202945055583123, 0.14709411288640353, 0.15536549770059047, 0.30856261674165014, 0.06868326711304348]
}
```

### Load it into PyTorch

Create `src/llm_lab/go_parity.py`. It reads the shapes from the JSON, builds a `float64` model, and copies every Go matrix into the matching layer using the table from Milestone 3:

```python
import json
from pathlib import Path

import torch

from .model import TinyDecoder

LAYERS = {
    "Embeddings": "embedding",
    "WQ": "query",
    "WK": "key",
    "WV": "value",
    "WO": "attention_output",
    "W1": "feed_forward_in",
    "W2": "feed_forward_out",
    "Output": "output",
}


def load_go_reference(path):
    reference = json.loads(Path(path).read_text())
    parameters = reference["parameters"]
    vocabulary, width = torch.tensor(parameters["Embeddings"]).shape
    hidden = len(parameters["W1"])
    model = TinyDecoder(vocabulary, width, hidden).double()
    with torch.no_grad():
        for go_name, layer_name in LAYERS.items():
            weight = getattr(model, layer_name).weight
            weight.copy_(torch.tensor(parameters[go_name], dtype=torch.float64))
    expected = {
        name: torch.tensor(reference[name], dtype=torch.float64)
        for name in ("states", "final_logits", "probabilities")
    }
    token_ids = torch.tensor([reference["token_ids"]])
    return model, token_ids, expected
```

Create `tests/test_go_parity.py`:

```python
from pathlib import Path

import pytest
import torch

from llm_lab.go_parity import load_go_reference

REFERENCE = Path(__file__).parent / "fixtures" / "demo_reference.json"


@pytest.mark.skipif(not REFERENCE.exists(), reason="run the Go exporter first")
def test_matches_go_demo_model():
    model, token_ids, expected = load_go_reference(REFERENCE)
    with torch.no_grad():
        states, _ = model.states(token_ids)
        logits = model.output(states[0, -1])
        probabilities = torch.softmax(logits, dim=-1)
    differences = {
        "states": (states[0] - expected["states"]).abs().max().item(),
        "final_logits": (logits - expected["final_logits"]).abs().max().item(),
        "probabilities": (probabilities - expected["probabilities"]).abs().max().item(),
    }
    print(differences)
    assert all(difference < 1e-12 for difference in differences.values())


@pytest.mark.skipif(not REFERENCE.exists(), reason="run the Go exporter first")
def test_future_token_does_not_change_earlier_states():
    model, _, _ = load_go_reference(REFERENCE)
    with torch.no_grad():
        first, _ = model.states(torch.tensor([[0, 2, 1]]))
        second, _ = model.states(torch.tensor([[0, 2, 4]]))
    assert torch.equal(first[0, :2], second[0, :2])
    assert not torch.equal(first[0, 2], second[0, 2])
```

Run with `-s` to see the measured differences:

```sh
PYTHONPATH=src pytest -q -s tests/test_go_parity.py
```

**Exit check:** the maximum absolute differences are on the order of \(10^{-16}\) (the measured run gave `states 1.1e-16`, `final_logits 5.6e-17`, `probabilities 1.4e-17`), well inside the \(10^{-12}\) tolerance. Both implementations use `float64`; the remaining difference is floating-point summation order. In `float32` expect around \(10^{-7}\) and use a tolerance near \(10^{-5}\).

### Diagnosing a mismatch

If parity fails, compare stage by stage instead of only the final probabilities. Add a temporary notebook cell that recomputes each stage from `model` and compare it with the same stage computed by hand from the JSON:

| Symptom | Likely cause |
|---|---|
| `states` differ at position 0 only by the position vector | `sinusoidal_positions` exponent or sin/cos order |
| all positions differ, position 0 included | a weight copied as its transpose, or wrong layer mapping |
| position 0 matches, later positions differ | mask direction (`diagonal=1` vs `diagonal=0`) or missing \(\sqrt{d_k}\) scale |
| states match, logits differ | `Output` matrix mapping, or logits taken from the wrong position |
| differences around \(10^{-7}\) | model or inputs left in `float32` |

## Milestone 7: greedy generation

**Reason:** generation is a loop around the model, not part of it. Keep it in its own file so decoding policy never changes model logits.

Create `src/llm_lab/generation.py`:

```python
import torch


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new_tokens, context_limit, eos_id=None):
    if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
        raise ValueError("prompt_ids must have shape [1, sequence]")
    if prompt_ids.size(1) == 0:
        raise ValueError("prompt must contain at least one token")
    if prompt_ids.size(1) > context_limit:
        raise ValueError(
            f"prompt has {prompt_ids.size(1)} tokens; context limit is {context_limit}"
        )
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        window = ids[:, -context_limit:]
        logits, _ = model(window)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return ids


@torch.no_grad()
def cached_logits(model, token_ids):
    model.eval()
    cache = None
    rows = []
    for position in range(token_ids.size(1)):
        logits, cache = model.step(token_ids[:, position : position + 1], position, cache)
        rows.append(logits)
    return torch.stack(rows, dim=1)
```

The function rejects an empty prompt and a prompt longer than `context_limit` instead of silently truncating it. During generation it explicitly keeps the last `context_limit` tokens, because sinusoidal positions are defined for any length but the model was only trained on positions below that limit. It recomputes the whole window every step; a KV cache comes later and must reproduce these logits.

Create `tests/test_generation.py`. A fake model that always predicts `id + 1` makes the expected output obvious, so the test checks the loop independently of training:

```python
import pytest
import torch
from torch import nn

from llm_lab.generation import greedy_generate


class NextIdModel(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.vocabulary = vocabulary

    def forward(self, token_ids):
        targets = (token_ids + 1) % self.vocabulary
        return nn.functional.one_hot(targets, self.vocabulary).float(), None


def test_greedy_follows_argmax():
    ids = greedy_generate(NextIdModel(5), torch.tensor([[0]]), 6, context_limit=4)
    assert ids.tolist() == [[0, 1, 2, 3, 4, 0, 1]]


def test_stops_on_eos():
    ids = greedy_generate(NextIdModel(5), torch.tensor([[0]]), 10, context_limit=4, eos_id=3)
    assert ids.tolist() == [[0, 1, 2, 3]]


def test_rejects_empty_and_oversized_prompts():
    with pytest.raises(ValueError):
        greedy_generate(NextIdModel(5), torch.empty(1, 0, dtype=torch.long), 1, context_limit=4)
    with pytest.raises(ValueError):
        greedy_generate(NextIdModel(5), torch.zeros(1, 5, dtype=torch.long), 1, context_limit=4)
```

**Exit check:** 3 tests pass, and the Milestone 5 experiment continues the cycle from `[5, 6]`.

## Milestone 8: checkpoints

**Reason:** a checkpoint is only useful if loading it reproduces the trained function exactly. Validate on load rather than trusting the file.

Create `src/llm_lab/checkpoint.py`:

```python
import hashlib
from pathlib import Path

import torch

from .model import TinyDecoder

FORMAT_VERSION = 1


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(path, model: TinyDecoder, metadata=None):
    path = Path(path)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": model.config(),
            "metadata": metadata or {},
            "state_dict": model.state_dict(),
        },
        path,
    )
    path.with_suffix(".sha256").write_text(_sha256(path) + "\n")


def load_checkpoint(path):
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text().strip()
    if _sha256(path) != expected:
        raise ValueError(f"{path} does not match its recorded SHA-256")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported format version {payload.get('format_version')}")
    model = TinyDecoder(**payload["config"])
    reference = model.state_dict()
    stored = payload["state_dict"]
    missing = reference.keys() - stored.keys()
    unexpected = stored.keys() - reference.keys()
    if missing or unexpected:
        raise ValueError(f"missing={sorted(missing)} unexpected={sorted(unexpected)}")
    for name, tensor in stored.items():
        if tensor.shape != reference[name].shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}; want {tuple(reference[name].shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values")
    model.load_state_dict(stored, strict=True)
    return model, payload["metadata"]
```

The file stores a format version, the model configuration, free-form metadata (seed, data source, step), and the state dictionary. A `.sha256` file is written next to it. Loading rejects a checksum mismatch, an unknown version, missing or unexpected tensors, wrong shapes, and NaN or infinite values, then uses `strict=True`. `weights_only=True` stops `torch.load` from running arbitrary pickled code.

Create `tests/test_checkpoint.py`:

```python
import pytest
import torch

from llm_lab.checkpoint import load_checkpoint, save_checkpoint
from llm_lab.model import TinyDecoder


def test_round_trip_reproduces_logits(tmp_path):
    torch.manual_seed(0)
    model = TinyDecoder(vocabulary=8, width=16, hidden=32)
    token_ids = torch.tensor([[0, 3, 5, 1]])
    save_checkpoint(tmp_path / "model.pt", model, {"seed": 0})
    loaded, metadata = load_checkpoint(tmp_path / "model.pt")
    assert metadata == {"seed": 0}
    assert torch.equal(model(token_ids)[0], loaded(token_ids)[0])


def test_rejects_corrupted_file(tmp_path):
    model = TinyDecoder(vocabulary=8, width=16, hidden=32)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(path)


def test_rejects_non_finite_weights(tmp_path):
    model = TinyDecoder(vocabulary=8, width=16, hidden=32)
    with torch.no_grad():
        model.output.weight[0, 0] = float("nan")
    save_checkpoint(tmp_path / "model.pt", model)
    with pytest.raises(ValueError, match="NaN"):
        load_checkpoint(tmp_path / "model.pt")
```

**Exit check:** 3 tests pass. The round-trip test compares logits with `torch.equal`, not a tolerance: saving and loading must be bit-exact.

At this point the suite has 21 passing tests (4 from Milestone 0, 3 from the gradient section, and 14 from Milestones 1–8).

## Milestone 9: multi-head attention

**Reason:** one head computes one weighting of the past per position. Several heads let different relations (for example "previous token" and "same token earlier") be tracked at the same time, at the same total cost.

With width \(d\) and \(h\) heads, each head works on its own slice of \(d/h\) coordinates of Q, K and V:

1. **Split:** `[batch, sequence, width]` → `[batch, heads, sequence, width/heads]`.
2. **Attend:** run `causal_attention` unchanged. It already works on any leading dimensions, and its \(\sqrt{d_k}\) now uses the head width.
3. **Merge:** move heads back and concatenate → `[batch, sequence, width]`.
4. **Mix:** the existing `attention_output` layer (\(W_O\)) combines information across heads.

Add to `src/llm_lab/attention.py`:

```python
def split_heads(x: torch.Tensor, heads: int):
    batch, length, width = x.shape
    if width % heads:
        raise ValueError(f"width {width} is not divisible by {heads} heads")
    return x.view(batch, length, heads, width // heads).transpose(1, 2)


def merge_heads(x: torch.Tensor):
    batch, heads, length, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_width)


def multi_head_causal_attention(q, k, v, heads: int):
    context, weights = causal_attention(
        split_heads(q, heads), split_heads(k, heads), split_heads(v, heads)
    )
    return merge_heads(context), weights


def attend_to_past(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    return torch.softmax(scores, dim=-1) @ v
```

`attend_to_past` is used by Milestone 10. In `model.py`, add a `heads=1` argument, store it in `config()`, and call `multi_head_causal_attention(q, k, v, self.heads)` instead of `causal_attention`. The complete file is in Milestone 10. With the default `heads=1` the Go parity test must still pass unchanged.

Create `tests/test_multi_head.py`:

```python
import torch

from llm_lab.attention import (
    causal_attention,
    merge_heads,
    multi_head_causal_attention,
    split_heads,
)


def test_split_and_merge_are_inverse():
    x = torch.randn(2, 5, 8)
    assert split_heads(x, 4).shape == (2, 4, 5, 2)
    assert torch.equal(merge_heads(split_heads(x, 4)), x)


def test_one_head_equals_single_head_attention():
    q, k, v = torch.randn(3, 2, 6, 8).unbind(0)
    expected, _ = causal_attention(q, k, v)
    actual, _ = multi_head_causal_attention(q, k, v, heads=1)
    assert torch.allclose(actual, expected)


def test_each_head_attends_on_its_own_slice():
    q, k, v = torch.randn(3, 1, 4, 6).unbind(0)
    output, weights = multi_head_causal_attention(q, k, v, heads=2)
    assert weights.shape == (1, 2, 4, 4)
    second, _ = causal_attention(q[..., 3:], k[..., 3:], v[..., 3:])
    assert torch.allclose(output[..., 3:], second)


def test_multi_head_is_still_causal():
    q, k, v = torch.randn(3, 1, 5, 8).unbind(0)
    changed = v.clone()
    changed[:, -1] += 100.0
    first, _ = multi_head_causal_attention(q, k, v, heads=4)
    second, _ = multi_head_causal_attention(q, k, changed, heads=4)
    assert torch.equal(first[:, :-1], second[:, :-1])
```

**Exit check:** 4 tests pass and `test_go_parity.py` still passes. The third test is the key one: with 2 heads, the second half of the output depends only on the second half of Q, K and V.

## Milestone 10: KV cache

**Reason:** `greedy_generate` recomputes every earlier position at every step, so generating \(n\) tokens costs about \(n^2\) position computations. A cache keeps each position's key and value rows, so a new token computes only its own row.

This is valid because of causality: the key and value of position \(j\) depend only on tokens \(0..j\), so they never change when tokens are appended. One generation step with a cache:

1. Embed the new token and add the position vector for its index.
2. Compute its \(q\), \(k\), \(v\) (one row each).
3. Append \(k\) and \(v\) to the cache along the sequence dimension.
4. Attend from the single \(q\) to *all* cached keys. No mask is needed, because everything in the cache is in the past.
5. Apply the residuals, the feed-forward layer, and the output layer to that one row.

The complete `src/llm_lab/model.py` after Milestones 9 and 10. `_finish` is shared by both paths so the residual order cannot drift between them:

```python
import torch
from torch import nn
from torch.nn import functional as F

from .attention import attend_to_past, merge_heads, multi_head_causal_attention, split_heads


def sinusoidal_positions(length: int, width: int, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    indices = torch.arange(width, device=device)
    exponent = (2 * torch.div(indices, 2, rounding_mode="floor")) / width
    angles = positions / (10000.0 ** exponent)
    result = torch.empty(length, width, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(angles[:, 0::2])
    result[:, 1::2] = torch.cos(angles[:, 1::2])
    return result


class TinyDecoder(nn.Module):
    def __init__(self, vocabulary, width, hidden, heads=1):
        super().__init__()
        self.vocabulary = vocabulary
        self.width = width
        self.hidden = hidden
        self.heads = heads
        self.embedding = nn.Embedding(vocabulary, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.feed_forward_in = nn.Linear(width, hidden, bias=False)
        self.feed_forward_out = nn.Linear(hidden, width, bias=False)
        self.output = nn.Linear(width, vocabulary, bias=False)

    def config(self):
        return {
            "vocabulary": self.vocabulary,
            "width": self.width,
            "hidden": self.hidden,
            "heads": self.heads,
        }

    def _finish(self, states, context):
        states = states + self.attention_output(context)
        return states + self.feed_forward_out(F.relu(self.feed_forward_in(states)))

    def states(self, token_ids):
        length = token_ids.size(-1)
        states = self.embedding(token_ids)
        states = states + sinusoidal_positions(
            length, self.width, states.device, states.dtype
        )
        q = self.query(states)
        k = self.key(states)
        v = self.value(states)
        context, weights = multi_head_causal_attention(q, k, v, self.heads)
        return self._finish(states, context), weights

    def forward(self, token_ids):
        states, weights = self.states(token_ids)
        return self.output(states), weights

    def step(self, token_id, position, cache=None):
        state = self.embedding(token_id)
        state = state + sinusoidal_positions(
            position + 1, self.width, state.device, state.dtype
        )[position]
        q = self.query(state)
        k = self.key(state)
        v = self.value(state)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
        context = merge_heads(
            attend_to_past(
                split_heads(q, self.heads),
                split_heads(k, self.heads),
                split_heads(v, self.heads),
            )
        )
        state = self._finish(state, context)
        return self.output(state)[:, -1], {"k": k, "v": v}
```

Add to `src/llm_lab/generation.py`. It feeds tokens one at a time through `step` and returns logits with the same shape as `model(token_ids)`:

```python
@torch.no_grad()
def cached_logits(model, token_ids):
    model.eval()
    cache = None
    rows = []
    for position in range(token_ids.size(1)):
        logits, cache = model.step(token_ids[:, position : position + 1], position, cache)
        rows.append(logits)
    return torch.stack(rows, dim=1)
```

Create `tests/test_kv_cache.py`:

```python
import torch

from llm_lab.generation import cached_logits
from llm_lab.model import TinyDecoder


def test_cached_logits_match_full_recompute():
    torch.manual_seed(0)
    model = TinyDecoder(vocabulary=11, width=16, hidden=32, heads=4)
    token_ids = torch.randint(0, 11, (1, 12))
    with torch.no_grad():
        full, _ = model(token_ids)
    cached = cached_logits(model, token_ids)
    assert cached.shape == full.shape
    assert (cached - full).abs().max().item() < 1e-6


def test_cache_grows_by_one_row_per_token():
    model = TinyDecoder(vocabulary=5, width=8, hidden=8, heads=2)
    cache = None
    with torch.no_grad():
        for position, token in enumerate([1, 3, 2]):
            _, cache = model.step(torch.tensor([[token]]), position, cache)
            assert cache["k"].shape == (1, position + 1, 8)
```

**Exit check:** 2 tests pass. Cached logits match full recomputation at *every* position within \(10^{-6}\) in `float32`, not just the final one. This cache is valid only while the sequence fits the context limit. Sliding the window would change every cached position index, so the cache must then be rebuilt.

## Milestone 11: a real tokenizer

**Reason:** until now token IDs were invented integers. A tokenizer is the exact, reversible map between text and IDs, and the model is tied to it: the same text must always produce the same IDs.

Start with a byte-level tokenizer. UTF-8 already turns any text into bytes 0–255, so it needs no training, never meets an unknown character, and round-trips exactly. Add one special ID, 256, for end-of-sequence.

Create `src/llm_lab/tokenizer.py`:

```python
class ByteTokenizer:
    eos_id = 256
    vocabulary = 257

    def encode(self, text: str, add_eos: bool = False):
        ids = list(text.encode("utf-8"))
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids):
        data = bytes(i for i in ids if i != self.eos_id)
        return data.decode("utf-8", errors="replace")
```

Create `tests/test_tokenizer.py`:

```python
import pytest

from llm_lab.tokenizer import ByteTokenizer

SAMPLES = ["", "Hello, world!", "  tabs\tand\nnewlines  ", "नमस्ते", "日本語", "naïve café 🙂"]


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_eos_is_outside_byte_range():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("hi", add_eos=True)
    assert ids == [104, 105, 256]
    assert tokenizer.decode(ids) == "hi"


def test_token_count_depends_on_script():
    tokenizer = ByteTokenizer()
    assert len(tokenizer.encode("hello")) == 5
    assert len(tokenizer.encode("नमस्ते")) == 18
    assert len(tokenizer.encode("日本語")) == 9
```

**Exit check:** 8 tests pass. The last test shows the cost: "hello" is 5 tokens, but "नमस्ते" (6 characters) is 18 and "日本語" (3 characters) is 9, because those characters take 3 bytes each in UTF-8. Longer sequences mean less text fits in the context and more compute per sentence. This is the fragmentation confounder mentioned in the [language-effect benchmark](notes/06-interpretation.md#how-to-benchmark-the-effect-of-language).

**Byte-pair encoding (BPE)** fixes this by learning merges on top of bytes:

1. Encode the training corpus as bytes.
2. Count every adjacent pair of IDs.
3. Replace the most frequent pair everywhere with a new ID (257, 258, …) and record the merge.
4. Repeat until the vocabulary reaches the target size.
5. To encode new text, apply the recorded merges in the same order. To decode, expand each ID back into its bytes.

Implement it as `BpeTokenizer` with the same `encode`/`decode` interface and run the same round-trip tests. Then compare token counts per language against `ByteTokenizer`. To train the model on text, set `vocabulary=tokenizer.vocabulary` in `TinyDecoder` and save the tokenizer's merges next to the checkpoint.

At this point the full suite should report `35 passed`.

## Later milestones

After the eleven milestones above pass, add one feature per branch or commit. Each row names the change and the test that must pass before the next row starts.

| # | Change | Where | Exit check |
|---|---|---|---|
| 12 | `nn.LayerNorm` before attention and before the feed-forward block (pre-norm) | `model.py` | Every position's normalized vector has mean ≈ 0 and variance ≈ 1; future-token and cache tests still pass |
| 13 | Replace `F.relu` with `F.gelu` | `model.py` | Rerun Milestone 5; record both loss curves in `experiments/runs/` |
| 14 | `layers` argument: stack blocks in an `nn.ModuleList`, one KV cache per layer | `model.py` | `layers=1` reproduces Go parity; cache test passes with `layers=3` |
| 15 | Small licensed text corpus with train/validation split and minibatches, using the Milestone 11 tokenizer | `training.py` | Validation loss reported every N steps; corpus source, license, and hash recorded in `result.json` |
| 16 | `temperature`, top-k, and nucleus sampling as functions applied to logits | `generation.py` | Fixed seed gives a fixed sample; temperature → 0 matches greedy; model logits are unchanged |
| 17 | Multilingual language-effect benchmark | `experiments/` | Follows the [benchmark protocol](notes/06-interpretation.md#how-to-benchmark-the-effect-of-language); reports token counts per language |
| 18 | Fine-tuning | `experiments/` | Before/after scores on a held-out set that existed before fine-tuning started |

At each milestone, retain a small test that can be checked by hand. The aim is to understand why the model works before increasing its size.
