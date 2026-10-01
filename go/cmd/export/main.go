// Command export writes DemoModel's parameters and reference outputs as JSON
// so another implementation can load the same weights and check parity.
package main

import (
	"encoding/json"
	"log"
	"os"

	"mathematics-llm-go/minillm"
)

type reference struct {
	Parameters    minillm.Model  `json:"parameters"`
	TokenIDs      []int          `json:"token_ids"`
	States        minillm.Matrix `json:"states"`
	FinalLogits   minillm.Vector `json:"final_logits"`
	Probabilities minillm.Vector `json:"probabilities"`
}

func main() {
	model := minillm.DemoModel()
	tokenIDs := []int{0, 2, 1}

	states, err := model.ForwardStates(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}
	logits, err := minillm.MatVec(model.Output, states[len(states)-1])
	if err != nil {
		log.Fatal(err)
	}
	probabilities, err := minillm.Softmax(logits)
	if err != nil {
		log.Fatal(err)
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(reference{model, tokenIDs, states, logits, probabilities}); err != nil {
		log.Fatal(err)
	}
}
