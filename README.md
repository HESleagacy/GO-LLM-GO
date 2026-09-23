# Mathematics of LLMs — Go Notes

An educational documentation site and a dependency-free Go forward-pass demonstration based on Joseph L. Breeden's *The Simple Mathematics of Large Language Models* (20-page source PDF referenced by section).

## What is included

- Topic-by-topic notes following the paper's 13-section progression.
- Equations translated into readable notation, with dimensions stated explicitly.
- Notes that distinguish what the paper says from corrections or implementation nuance.
- A small causal self-attention forward pass in Go, built from primitive matrix/vector operations.
- Tests for stable softmax, token-level causal masking, and tensor shape checks.
- A hand-checkable end-to-end example, local SVG diagrams, notation reference, and glossary.
- An inference-readiness roadmap for turning the demo into a trained small language model.

## Run the Go lab

```sh
cd go
go test ./...
go run ./cmd/demo
```

The Go code is an inspectable teaching model. It does **not** train a language model, include a tokenizer, or claim to reproduce a pretrained LLM. Its initialized weights are deterministic demonstration values, so generated probabilities are not meaningful language predictions.

## Preview the docs

The package includes a prebuilt `site/` directory with local MathJax assets, so the equations do not depend on a third-party CDN. Serve it locally with:

```sh
cd site
python -m http.server 8000
```

Then open `http://localhost:8000`. To edit the source docs, install MkDocs Material and the math extensions, then run:

```sh
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open the local address printed by MkDocs. To build static HTML, run `mkdocs build --strict`; output goes to `site/`.

## Source

See [Source and scope](docs/source-notes.md) for the exact source identification, citation convention, and technical caveats.
