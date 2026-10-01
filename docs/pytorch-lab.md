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

The finished layout after Milestone 8 is:

```text
llm-learning-lab/
├── src/llm_lab/
│   ├── __init__.py
│   ├── attention.py        Milestone 1
│   ├── model.py            Milestones 2–3
│   ├── training.py         Milestones 4–5
│   ├── go_parity.py        Milestone 6
│   ├── generation.py       Milestone 7
│   └── checkpoint.py       Milestone 8
├── tests/
│   ├── fixtures/demo_reference.json
│   ├── test_attention.py
│   ├── test_positions.py
│   ├── test_training.py
│   ├── test_go_parity.py
│   ├── test_generation.py
│   └── test_checkpoint.py
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

At this point the full suite should report `14 passed`.

## Later milestones

After the eight milestones above pass, add one feature per branch or commit. Each row names the change and the test that must pass before the next row starts.

| # | Change | Where | Exit check |
|---|---|---|---|
| 9 | `nn.LayerNorm` before attention and before the feed-forward block (pre-norm) | `model.py` | Every position's normalized vector has mean ≈ 0 and variance ≈ 1; future-token test still passes |
| 10 | Replace `F.relu` with `F.gelu` | `model.py` | Rerun Milestone 5; record both loss curves in `experiments/runs/` |
| 11 | `heads` argument: split Q/K/V into `[batch, heads, sequence, width/heads]`, attend per head, concatenate, apply `attention_output` | `attention.py`, `model.py` | With `heads=1` the output equals the single-head model exactly; with `heads>1` the causal test still passes |
| 12 | `layers` argument: stack blocks in an `nn.ModuleList` | `model.py` | `layers=1` reproduces Milestone 6 parity |
| 13 | Byte-level or BPE tokenizer with a saved vocabulary file | new `tokenizer.py` | `decode(encode(text)) == text` for ASCII, whitespace, punctuation, and non-ASCII samples |
| 14 | Small licensed text corpus with train/validation split and minibatches | `training.py` | Validation loss reported every N steps; corpus source, license, and hash recorded in `result.json` |
| 15 | `temperature`, top-k, and nucleus sampling as functions applied to logits | `generation.py` | Fixed seed gives a fixed sample; temperature → 0 matches greedy; model logits are unchanged |
| 16 | KV cache: store K and V per layer and feed only the new token | `generation.py`, `model.py` | Cached and uncached logits match at every generated position within \(10^{-6}\) in `float32` |
| 17 | Multilingual language-effect benchmark | `experiments/` | Follows the [benchmark protocol](notes/06-interpretation.md#how-to-benchmark-the-effect-of-language); reports token counts per language |
| 18 | Fine-tuning | `experiments/` | Before/after scores on a held-out set that existed before fine-tuning started |

At each milestone, retain a small test that can be checked by hand. The aim is to understand why the model works before increasing its size.
