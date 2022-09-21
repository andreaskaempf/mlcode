// Unit tests for linear regression

package main

import (
	//"math"
	"testing"
)

// Test complete multivariate linear regression example
func TestLinearRegression(t *testing.T) {

	// Read data into matrix, separate X and Y
	m, _ := readMatrixCSV("data/pizza_3_vars.txt")
	X := extractCols(m, 0, 2) // all cols except last
	Y := extractCols(m, 3, 3) // just the last col

	// Train linear regression model, test coefficients
	w := trainLinRegr(X, Y, .001, .001, false)
	r, c := w.Dims()
	if r != 3 || c != 1 {
		t.Error("Linear regression failed: write size returned")
	} else if !(close(w.At(0, 0), 1.1375) && close(w.At(1, 0), 0.1834) && close(w.At(2, 0), 3.0191)) {
		t.Error("Linear regression failed: wrong coefficients")
	}
}

