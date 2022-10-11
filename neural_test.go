// Testing for simple neural network code

package mlcode

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Test neural network functions
func TestNeuralNet(t *testing.T) {

	// Test softmax function
	input := mat.NewDense(2, 3, []float64{0.3, 0.8, 0.2, 0.1, 0.9, 0.1})
	expect := mat.NewDense(2, 3, []float64{0.2814, 0.4640, 0.2546, 0.2367, 0.5267, 0.2367})
	res := SoftMax(input)
	if !mat.EqualApprox(res, expect, .001) {
		t.Error("Softmax failed")
	}

}
