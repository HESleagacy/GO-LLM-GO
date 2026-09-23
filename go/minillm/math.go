package minillm

import (
	"errors"
	"fmt"
	"math"
)

type Vector []float64
type Matrix []Vector // Row-major: each Vector is one row.

func validateRectangular(m Matrix) (rows, cols int, err error) {
	if len(m) == 0 || len(m[0]) == 0 {
		return 0, 0, errors.New("matrix must have at least one row and column")
	}
	cols = len(m[0])
	for i, row := range m {
		if len(row) != cols {
			return 0, 0, fmt.Errorf("matrix row %d has width %d; want %d", i, len(row), cols)
		}
	}
	return len(m), cols, nil
}

func Dot(a, b Vector) (float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("dot product needs equal non-empty vectors; got %d and %d", len(a), len(b))
	}
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// MatVec computes Mx where M is stored row-major.
func MatVec(m Matrix, x Vector) (Vector, error) {
	_, cols, err := validateRectangular(m)
	if err != nil {
		return nil, err
	}
	if len(x) != cols {
		return nil, fmt.Errorf("matrix width %d does not match vector width %d", cols, len(x))
	}
	out := make(Vector, len(m))
	for i, row := range m {
		for j, value := range row {
			out[i] += value * x[j]
		}
	}
	return out, nil
}

// Softmax uses max subtraction for numerical stability.
func Softmax(logits Vector) (Vector, error) {
	if len(logits) == 0 {
		return nil, errors.New("softmax needs at least one logit")
	}
	maxLogit := logits[0]
	for _, x := range logits {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, errors.New("softmax logits must be finite")
		}
		if x > maxLogit {
			maxLogit = x
		}
	}
	out := make(Vector, len(logits))
	var total float64
	for i, x := range logits {
		out[i] = math.Exp(x - maxLogit)
		total += out[i]
	}
	for i := range out {
		out[i] /= total
	}
	return out, nil
}

func add(a, b Vector) (Vector, error) {
	if len(a) == 0 || len(a) != len(b) {
		return nil, fmt.Errorf("vector addition needs equal non-empty widths; got %d and %d", len(a), len(b))
	}
	out := make(Vector, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out, nil
}

func relu(x Vector) Vector {
	out := make(Vector, len(x))
	for i, value := range x {
		if value > 0 {
			out[i] = value
		}
	}
	return out
}

func sinusoidalPosition(position, width int) Vector {
	pos := make(Vector, width)
	for i := 0; i < width; i++ {
		exponent := float64(2*(i/2)) / float64(width)
		angle := float64(position) / math.Pow(10000, exponent)
		if i%2 == 0 {
			pos[i] = math.Sin(angle)
		} else {
			pos[i] = math.Cos(angle)
		}
	}
	return pos
}
