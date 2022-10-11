// Unit tests for logistic regression

package mlcode

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Test logistic regression calculations
func TestLogRegr(t *testing.T) {

	// Test sigmoid function
	// TODO: test negative values?
	try := []float64{0, .25, .50, 1}
	expect := []float64{.5, .562, .622, .731}
	for i := 0; i < len(try); i++ {
		if !close(Sigmoid(try[i]), expect[i]) {
			fmt.Printf("sigmoid(%f) produced %f instead of %f\n", try[i], Sigmoid(try[i]), expect[i])
			t.Error("Sigmoid failed")
		}
	}

	// Create a model
	m := LogisticRegression{}
	m.lr = 0.001

	// Set up input values
	X := mat.NewDense(5, 3, []float64{13, 26, 9, 2, 14, 6, 14, 20, 3, 23, 25, 9, 13, 24, 8})
	m.w = mat.NewDense(3, 1, []float64{0.0346, 0.0396, 0.00977})

	// Test forward()
	expect1 := mat.NewDense(5, 1, []float64{.827, .664, .787, .867, .814})
	fwd := m.Forward(X)
	if !matSame(fwd, expect1) {
		fmt.Println("X =")
		MatPrint(X)
		fmt.Println("w =")
		MatPrint(m.w)
		fmt.Println("forward(X) =")
		MatPrint(fwd)
		fmt.Println("expected:")
		MatPrint(expect1)
		t.Error("Forward failed")
	}

	// Test classify, based on previous forward values
	cls := m.Classify(X)
	expect2 := mat.NewDense(5, 1, []float64{1, 1, 1, 1, 1})
	if !matSame(cls, expect2) {
		t.Error("Classify failed")
	}

	// Test loss
	m.w = mat.NewDense(3, 1, []float64{0.03257874, 0.03791611, 0.00951759})
	Y := mat.NewDense(5, 1, []float64{1, 0, 1, 1, 1})
	expect3 := 0.38037
	lss := m.Loss(X, Y)
	if !close(lss, expect3) {
		fmt.Printf("Loss was %f instead of %f\n", lss, expect3)
		fmt.Println("X =")
		MatPrint(X)
		fmt.Println("Y =")
		MatPrint(Y)
		fmt.Println("w =")
		MatPrint(m.w)
		t.Error("Loss failed")
	}

	// Test gradient
	grads := m.Gradient(X, Y)
	expect4 := mat.NewDense(3, 1, []float64{-2.01265296, -1.66852372, -0.24798428})
	if !matSame(grads, expect4) {
		t.Error("gradient failed")
	}

	// Test adjustment of weights using gradient (embedded inside training
	// function, so replicate code here)
	grads.Scale(m.lr, grads)
	m.w.Sub(m.w, grads)
	expect5 := mat.NewDense(3, 1, []float64{0.03459139, 0.03958463, 0.00976558})
	if !matSame(m.w, expect5) {
		t.Error("gradient failed")
	}
}
