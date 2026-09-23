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

func TestForwardStatesDoNotReadFutureTokens(t *testing.T) {
	model := DemoModel()
	withoutFuture, err := model.ForwardStates([]int{0, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	withDifferentFuture, err := model.ForwardStates([]int{0, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	for position := 0; position < 2; position++ {
		for dimension := range withoutFuture[position] {
			if math.Abs(withoutFuture[position][dimension]-withDifferentFuture[position][dimension]) > 1e-12 {
				t.Fatalf("future token changed state at position %d, dimension %d", position, dimension)
			}
		}
	}
	if math.Abs(withoutFuture[2][0]-withDifferentFuture[2][0]) < 1e-12 && math.Abs(withoutFuture[2][1]-withDifferentFuture[2][1]) < 1e-12 {
		t.Fatal("expected changed future token to affect its own position")
	}
}
