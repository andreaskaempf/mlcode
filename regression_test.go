// Unit tests for linear regression

package main

import (
	"math"
	"testing"
)

// Test complete multivariate linear regression example
func TestLinearRegression(t *testing.T) {

	// Read data into matrix, separate X and Y
	m, _ := readMatrixCSV("data/pizza_3_vars.txt")
	X := extractCols(m, 0, 2) // all cols except last
	Y := extractCols(m, 3, 3) // just the last col

	// Train linear regression model, test coefficients
	w := trainRegression(X, Y, .001, .001, false)
	r, c := w.Dims()
	if r != 3 || c != 1 {
		t.Error("Linear regression failed: write size returned")
	} else if !(close(w.At(0, 0), 1.1375) && close(w.At(1, 0), 0.1834) && close(w.At(2, 0), 3.0191)) {
		t.Error("Linear regression failed: wrong coefficients")
	}
}

// Test whether two lists of floats are the same
func same(A, B []float64) bool {
	if len(A) != len(B) {
		return false
	}
	for i := 0; i < len(A); i++ {
		if !close(A[i], B[i]) {
			return false
		}
	}
	return true
}

// Test whether two numbers are close
func close(a, b float64) bool {
	if a == 0 && b == 0 {
		return true
	}
	return math.Abs(a-b)/((a+b)/2.0) < .001
}
