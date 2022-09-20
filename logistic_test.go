// Unit tests for logistic regression

package main

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Test sigmoid function
// TODO: test negative values?
func TestSigmoid(t *testing.T) {
	try := []float64{0, .25, .50, 1}
	expect := []float64{.5, .562, .622, .731}
	for i := 0; i < len(try); i++ {
		if !close(sigmoid(try[i]), expect[i]) {
			fmt.Printf("sigmoid(%f) produced %f instead of %f\n", try[i], sigmoid(try[i]), expect[i])
			t.Error("Sigmoid failed")
		}
	}
}

// Test forward, classify, loss calculations
func TestLogRegr(t *testing.T) {

	// Set up input values
	X := mat.NewDense(5, 3, []float64{13, 26, 9, 2, 14, 6, 14, 20, 3, 23, 25, 9, 13, 24, 8})
	w := mat.NewDense(3, 1, []float64{0.0346, 0.0396, 0.00977})

	// Test forward()
	expect1 := mat.NewDense(5, 1, []float64{.827, .664, .787, .867, .814})
	fwd := forward(X, w)
	if !matSame(fwd, expect1) {
		fmt.Println("X =")
		matPrint(X)
		fmt.Println("w =")
		matPrint(w)
		fmt.Println("forward(X, w) =")
		matPrint(fwd)
		fmt.Println("expected:")
		matPrint(expect1)
		t.Error("Forward failed")
	}

	// Test classify, based on previous forward values
	cls := classify(X, w)
	expect2 := mat.NewDense(5, 1, []float64{1, 1, 1, 1, 1})
	if !matSame(cls, expect2) {
		t.Error("Classify failed")
	}

	// Test loss
	w1 := mat.NewDense(3, 1, []float64{0.03257874, 0.03791611, 0.00951759})
	Y := mat.NewDense(5, 1, []float64{1, 0, 1, 1, 1})
	expect3 := 0.38037
	lss := lossLogRegr(X, Y, w1)
	if !close(lss, expect3) {
		fmt.Printf("Loss was %f instead of %f\n", lss, expect3)
        fmt.Println("X =")
        matPrint(X)
        fmt.Println("Y =")
        matPrint(Y)
        fmt.Println("w =")
        matPrint(w1)
		t.Error("Loss failed")
	}

    // Test gradient
    grads := gradientLogReg(X, Y, w1)
    expect4 := mat.NewDense(3, 1, []float64{-2.01265296, -1.66852372, -0.24798428})
	if !matSame(grads, expect4) {
		t.Error("gradient failed")
    }

    // Test adjustment of weights using gradient (embedded inside training
    // function, so replicate code here)
    lr := 0.001 
    grads.Scale(lr, grads)
    w1.Sub(w, grads)
    expect5 := mat.NewDense(3, 1, []float64{0.03459139, 0.03958463, 0.00976558})
	if !matSame(w, expect5) {
		t.Error("gradient failed")
    }
}

// Another test of gradient function
func TestLogGrad(t *testing.T) {
    X := mat.NewDense(3, 3, []float64{1,  2,  3, 1.2, 1.9, 3.2, 0.9, 1.8, 3.1})
    Y := mat.NewDense(3, 1, []float64{1, 0, 1})
    w := mat.NewDense(3, 1, []float64{0.1, 0.2, 0.3})
    expect := mat.NewDense(3, 1, []float64{0.19837167, 0.26148789, 0.46010944})
    g := gradientLogReg(X, Y, w)
	if !matSame(g, expect) {
        fmt.Println("Wrong result:")
        matPrint(g)
        fmt.Println("instead of")
        matPrint(expect)
		t.Error("Additional gradient test failed")
    }
}
