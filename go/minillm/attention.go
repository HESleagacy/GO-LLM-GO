package minillm

import (
	"fmt"
	"math"
)

// CausalAttention computes scaled dot-product attention with a future-token mask.
// q and k have shape [sequence, keyWidth], v has shape [sequence, valueWidth].
// The result has shape [sequence, valueWidth].
func CausalAttention(q, k, v Matrix) (Matrix, error) {
	qRows, qWidth, err := validateRectangular(q)
	if err != nil {
		return nil, fmt.Errorf("queries: %w", err)
	}
	kRows, kWidth, err := validateRectangular(k)
	if err != nil {
		return nil, fmt.Errorf("keys: %w", err)
	}
	vRows, vWidth, err := validateRectangular(v)
	if err != nil {
		return nil, fmt.Errorf("values: %w", err)
	}
	if qRows != kRows || qRows != vRows {
		return nil, fmt.Errorf("sequence lengths differ: queries=%d keys=%d values=%d", qRows, kRows, vRows)
	}
	if qWidth != kWidth {
		return nil, fmt.Errorf("query width %d does not match key width %d", qWidth, kWidth)
	}

	sequence := qRows
	out := make(Matrix, sequence)
	scale := math.Sqrt(float64(qWidth))
	for i := 0; i < sequence; i++ {
		scores := make(Vector, i+1) // causal mask: only j <= i is materialized
		for j := 0; j <= i; j++ {
			score, _ := Dot(q[i], k[j]) // widths were validated above
			scores[j] = score / scale
		}
		weights, err := Softmax(scores)
		if err != nil {
			return nil, fmt.Errorf("softmax at position %d: %w", i, err)
		}
		out[i] = make(Vector, vWidth)
		for j, weight := range weights {
			for c, value := range v[j] {
				out[i][c] += weight * value
			}
		}
	}
	return out, nil
}
