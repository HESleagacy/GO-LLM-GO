package minillm

import (
	"math"
	"testing"
)

func TestSoftmaxStableAndNormalized(t *testing.T) {
	got, err := Softmax(Vector{1000, 1001, 999})
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, value := range got {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("non-finite probability: %v", value)
		}
		sum += value
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("probabilities sum to %.16f, want 1", sum)
	}
}

func TestCausalAttentionDoesNotReadFutureValues(t *testing.T) {
	q := Matrix{{1, 0}, {0, 1}, {1, 1}}
	k := Matrix{{1, 0}, {0, 1}, {1, 1}}
	v1 := Matrix{{2, 0}, {0, 4}, {10, 10}}
	v2 := Matrix{{2, 0}, {0, 4}, {-1000, 1000}}

	a, err := CausalAttention(q, k, v1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := CausalAttention(q, k, v2)
	if err != nil {
		t.Fatal(err)
	}
	for position := 0; position < 2; position++ {
		for dim := range a[position] {
			if math.Abs(a[position][dim]-b[position][dim]) > 1e-12 {
				t.Fatalf("future value changed output at position %d, dimension %d", position, dim)
			}
		}
	}
	if a[2][0] == b[2][0] && a[2][1] == b[2][1] {
		t.Fatal("expected final position to be able to use the changed value")
	}
}

func TestCausalAttentionRejectsMismatchedShapes(t *testing.T) {
	_, err := CausalAttention(Matrix{{1, 0}}, Matrix{{1}}, Matrix{{1, 2}})
	if err == nil {
		t.Fatal("expected key/query width mismatch error")
	}
}

func TestMatVecRejectsRaggedMatrix(t *testing.T) {
	_, err := MatVec(Matrix{{1, 2}, {3}}, Vector{1, 1})
	if err == nil {
		t.Fatal("expected ragged matrix error")
	}
}
