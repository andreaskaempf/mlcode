// Unit tests for multi-class logistic regression

package regression

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
	"mlcode/utils"
)

// Test logistic regression calculations
func TestMultiLogRegr(t *testing.T) {

	// Create a model
	m := MultiLogRegression{LR: .0001}

	// Set up input values
	X := mat.NewDense(5, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
	Y := mat.NewDense(5, 2, []float64{1, 0, 1, 0, 0, 1, 1, 0, 0, 1})
	m.w = mat.NewDense(3, 2, []float64{0.01, 0.02, 0.03, 0.04, 0.05, 0.06})

	// Uncomment these lines during debugging
	/*fmt.Println("X =")
	matPrint(X)
	fmt.Println("Y =")
	matPrint(Y)
	fmt.Println("w =")
	matPrint(m.w)*/

	// Test forward(X, w)
	expect1 := mat.NewDense(5, 2, []float64{0.5548, 0.5695, 0.6201, 0.6548,
		0.6814, 0.7311, 0.7369, 0.7957, 0.7858, 0.8481})
	fwd := m.Forward(X)
	if !utils.MatSame(fwd, expect1) {
		fmt.Println("forward(X) =")
		utils.MatPrint(fwd)
		fmt.Println("expected:")
		utils.MatPrint(expect1)
		t.Error("Forward failed")
	}

	// Test gradient(X, Y, w)
	expect2 := mat.NewDense(3, 2, []float64{2.0779, 1.4579, 2.1537, 1.7777, 2.2295, 2.0975})
	grad := m.Gradient(X, Y)
	if !utils.MatSame(grad, expect2) {
		fmt.Println("gradient(X, Y, w) =")
		utils.MatPrint(grad)
		fmt.Println("expected:")
		utils.MatPrint(expect2)
		t.Error("Gradient failed")
	}

	// Test scaling of gradient by learning rate
	lr := .01
	grad.Scale(lr, grad)
	expect3 := mat.NewDense(3, 2, []float64{0.0208, .0146, .0215, 0.0178, .0223, .0210})
	if !mat.EqualApprox(grad, expect3, .001) {
		fmt.Println("scaled gradient =")
		utils.MatPrint(grad)
		fmt.Println("expected:")
		utils.MatPrint(expect3)
		t.Error("Scaling failed")
	}

	// Test adjustment of weights
	m.w.Sub(m.w, grad)
	expect4 := mat.NewDense(3, 2, []float64{-0.01077939, 0.00542152, 0.00846263, 0.02222302, 0.02770465, 0.03902453})
	if !utils.MatSame(m.w, expect4) {
		fmt.Println("adjusted w =")
		utils.MatPrint(m.w)
		fmt.Println("expected:")
		utils.MatPrint(expect4)
		t.Error("Adjustment failed")
	}

	// Test loss function
	l := m.Loss(X, Y)
	expect5 := 1.4269
	if !utils.Close(l, expect5) {
		fmt.Printf("Loss failed: got %f, expected %f\n", l, expect5)
	}
}
