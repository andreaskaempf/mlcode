// Unit tests for linear regression

package mlcode

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Test complete multivariate linear regression example
func TestLinearRegression(t *testing.T) {

	// Create test data, and expected output
	X := mat.NewDense(5, 3, []float64{13, 26, 9, 2, 14, 6, 14, 20, 3, 23, 25, 9, 13, 24, 8})
	Y := mat.NewDense(5, 1, []float64{44, 23, 28, 60, 42})
	expect := mat.NewDense(3, 1, []float64{1.3586, -0.02001, 3.1468})

	// Create and train linear regression model, check if coefficients match
	m := LinearRegression{}
	m.train(X, Y)
	if !matSame(m.w, expect) {
		fmt.Println("Linear regression: got")
		matPrint(m.w)
		fmt.Println("instead of")
		matPrint(expect)
		t.Error("Linear regression failed")
	}
}
