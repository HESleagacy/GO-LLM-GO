package minillm

import (
	"math"
	"testing"
)

func TestDemoModelReturnsVocabularyDistribution(t *testing.T) {
	probabilities, err := DemoModel().Forward([]int{0, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(probabilities) != 5 {
		t.Fatalf("got %d vocabulary probabilities, want 5", len(probabilities))
	}
	var sum float64
	for _, p := range probabilities {
		if p < 0 || math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("invalid probability %v", p)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("probabilities sum to %.16f, want 1", sum)
	}
}

func TestForwardRejectsInvalidTokenID(t *testing.T) {
	_, err := DemoModel().Forward([]int{0, 99})
	if err == nil {
		t.Fatal("expected invalid token ID error")
	}
}
