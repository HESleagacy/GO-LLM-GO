package main

import (
	"fmt"
	"log"

	"mathematics-llm-go/minillm"
)

func main() {
	model := minillm.DemoModel()
	probabilities, err := model.Forward([]int{0, 2, 1})
	if err != nil {
		log.Fatal(err)
	}

	tokens := []string{"<next-0>", "<next-1>", "<next-2>", "<next-3>", "<next-4>"}
	var total float64
	fmt.Println("Toy next-token distribution (fixed demonstration weights):")
	for i, probability := range probabilities {
		fmt.Printf("  %-10s %.6f\n", tokens[i], probability)
		total += probability
	}
	fmt.Printf("sum: %.6f\n", total)
}
