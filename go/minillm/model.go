package minillm

import "fmt"

// Model is a deliberately small, single-head causal decoder block.
// Matrices are explicit so the math remains visible.
type Model struct {
	Embeddings Matrix // [vocabulary, width]
	WQ, WK, WV Matrix // projections; this demo uses square [width, width] matrices
	WO         Matrix // [width, width] attention output projection
	W1, W2     Matrix // feed-forward projections: [hidden, width], [width, hidden]
	Output     Matrix // [vocabulary, width]
}

// Forward returns next-token probabilities for the final token in tokenIDs.
func (m Model) Forward(tokenIDs []int) (Vector, error) {
	states, err := m.ForwardStates(tokenIDs)
	if err != nil {
		return nil, err
	}
	logits, err := MatVec(m.Output, states[len(states)-1])
	if err != nil {
		return nil, err
	}
	return Softmax(logits)
}

// ForwardStates returns the final per-position states before vocabulary output.
// It is exposed for teaching and invariant tests; it is not a cache.
func (m Model) ForwardStates(tokenIDs []int) (Matrix, error) {
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("input sequence must not be empty")
	}
	vocab, width, err := validateRectangular(m.Embeddings)
	if err != nil {
		return nil, fmt.Errorf("embeddings: %w", err)
	}
	if _, outWidth, err := validateRectangular(m.Output); err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	} else if outWidth != width || len(m.Output) != vocab {
		return nil, fmt.Errorf("output projection must have shape [%d,%d]", vocab, width)
	}
	for name, matrix := range map[string]Matrix{"WQ": m.WQ, "WK": m.WK, "WV": m.WV, "WO": m.WO} {
		rows, cols, err := validateRectangular(matrix)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if rows != width || cols != width {
			return nil, fmt.Errorf("%s must have shape [%d,%d]", name, width, width)
		}
	}
	hidden, w1Width, err := validateRectangular(m.W1)
	if err != nil {
		return nil, fmt.Errorf("W1: %w", err)
	}
	w2Rows, w2Width, err := validateRectangular(m.W2)
	if err != nil {
		return nil, fmt.Errorf("W2: %w", err)
	}
	if w1Width != width || w2Rows != width || w2Width != hidden {
		return nil, fmt.Errorf("feed-forward matrices must have shapes [hidden,%d] and [%d,hidden]", width, width)
	}

	states := make(Matrix, len(tokenIDs))
	for i, id := range tokenIDs {
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("token ID %d at position %d is outside vocabulary [0,%d)", id, i, vocab)
		}
		states[i], _ = add(m.Embeddings[id], sinusoidalPosition(i, width))
	}

	queries, keys, values := make(Matrix, len(states)), make(Matrix, len(states)), make(Matrix, len(states))
	for i, state := range states {
		queries[i], _ = MatVec(m.WQ, state)
		keys[i], _ = MatVec(m.WK, state)
		values[i], _ = MatVec(m.WV, state)
	}
	context, err := CausalAttention(queries, keys, values)
	if err != nil {
		return nil, err
	}

	for i := range states {
		projected, _ := MatVec(m.WO, context[i])
		states[i], _ = add(states[i], projected) // attention residual
		inner, _ := MatVec(m.W1, states[i])
		inner = relu(inner)
		outer, _ := MatVec(m.W2, inner)
		states[i], _ = add(states[i], outer) // feed-forward residual
	}

	return states, nil
}

// DemoModel returns fixed toy parameters. They make the example reproducible,
// not linguistically useful.
func DemoModel() Model {
	return Model{
		Embeddings: Matrix{
			{0.20, 0.10, -0.10, 0.00},
			{0.00, 0.30, 0.10, -0.20},
			{-0.20, 0.10, 0.25, 0.15},
			{0.15, -0.25, 0.10, 0.20},
			{-0.10, -0.10, 0.20, 0.30},
		},
		WQ: Matrix{{0.8, 0.1, 0.0, 0.0}, {0.0, 0.7, 0.1, 0.0}, {0.0, 0.0, 0.9, 0.1}, {0.1, 0.0, 0.0, 0.8}},
		WK: Matrix{{0.7, 0.0, 0.1, 0.0}, {0.1, 0.8, 0.0, 0.0}, {0.0, 0.1, 0.7, 0.0}, {0.0, 0.0, 0.1, 0.9}},
		WV: Matrix{{0.9, 0.0, 0.0, 0.1}, {0.0, 0.8, 0.1, 0.0}, {0.1, 0.0, 0.8, 0.0}, {0.0, 0.1, 0.0, 0.7}},
		WO: Matrix{{0.9, 0.0, 0.0, 0.0}, {0.0, 0.9, 0.0, 0.0}, {0.0, 0.0, 0.9, 0.0}, {0.0, 0.0, 0.0, 0.9}},
		W1: Matrix{{0.5, 0.1, 0.0, 0.0}, {0.0, 0.5, 0.1, 0.0}, {0.0, 0.0, 0.5, 0.1}, {0.1, 0.0, 0.0, 0.5}, {0.2, 0.1, 0.2, 0.1}, {0.1, 0.2, 0.1, 0.2}},
		W2: Matrix{{0.4, 0.0, 0.0, 0.1, 0.2, 0.1}, {0.0, 0.4, 0.1, 0.0, 0.1, 0.2}, {0.1, 0.0, 0.4, 0.0, 0.2, 0.1}, {0.0, 0.1, 0.0, 0.4, 0.1, 0.2}},
		Output: Matrix{
			{0.6, 0.0, 0.1, 0.0},
			{0.0, 0.6, 0.0, 0.1},
			{0.1, 0.0, 0.6, 0.0},
			{0.0, 0.1, 0.0, 0.6},
			{-0.3, -0.2, 0.1, 0.2},
		},
	}
}
