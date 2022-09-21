// Linear Regression
//
// Based on Python implementation in Chapters 2-4 of "Programming Machine
// Learning" by Paolo Perrotta.
//
// Sample usage (also see unit test file):
//	m, _ := readMatrixCSV("data/pizza_3_vars.txt")
//	X := extractCols(m, 0, 2) // all cols except last
//	Y := extractCols(m, 3, 3) // just the last col
//	w := trainLinRegr(X, Y, .001, .001, true)
//	matPrint(w) // prints final coefficients

package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Train linear regression model using gradient descent on coeffients,
// until loss stops improving by at least the tolerance. Returns vector
// of coefficients.
func trainLinRegr(X, Y *mat.Dense, lr, tol float64, verbose bool) *mat.Dense {

	// Initialize weights/coefficients to zero (a column vector, with length =
	// number of columns in X)
	_, c := X.Dims()
	w := mat.NewDense(c, 1, nil) // Python: np.zeros((X.shape[1], 1))

	// Initialize the previous loss, so we can detect when we are converging
	prevLoss := 0.0

	// Iterate until tolerance is low
	for i := 0; i < 1000; i++ { // maximum iterations, increase if necessary

		// Calculate loss using current weights, stop if no more improvement
		// from last iteration
		l := lossLinRegr(X, Y, w)
		if math.IsNaN(l) {
			fmt.Println("Loss cannot be computed")
			break
		}
		if verbose {
			fmt.Printf("Iteration %d: loss = %f\n", i, l)
		}
		if i > 0 && math.Abs(prevLoss-l) < tol {
			if verbose {
				fmt.Println("Solution found")
			}
			return w
		}

		// Remember loss for next iteration
		prevLoss = l

		// Adjust the weights (coefficients) using the gradients
		// Python: w -= gradient(X, Y, w) * lr
		grads := gradientLinReg(X, Y, w)
		grads.Scale(lr, grads)
		w.Sub(w, grads)
	}

	// If no solution found, return nil pointer
	return nil
}

// Predict Y values (one column), given X values (one column per variable) and
// coefficients (vector of values, one per X column)
// Python: return np.matmul(X, w)
func predictLinRegr(X, w *mat.Dense) *mat.Dense {
	xr, _ := X.Dims()
	res := mat.NewDense(xr, 1, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X, w)
	return res
}

// Calculate the mean squared difference between predicted and actual values,
// for linear regression
// Python: np.average((predict(X, w) - Y) ** 2)
func lossLinRegr(X, Y, w *mat.Dense) float64 {

	// Get differences of predictions vs. actual
	deltas := predictLinRegr(X, w)
	deltas.Sub(deltas, Y)

	// Compute the average of squared deltas
	var tot float64
	rows, _ := deltas.Dims()
	for i := 0; i < rows; i++ {
		n := deltas.At(i, 0)
		tot += n * n
	}
	return tot / float64(rows)
}

// Compute the gradient for linear regression
// Python: return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]
func gradientLinReg(X, Y, w *mat.Dense) *mat.Dense {

	// Get differences of predictions vs. actual
	// Python: (predict(X, w) - Y))
	deltas := predictLinRegr(X, w)
	deltas.Sub(deltas, Y)

	// Multiply transposed X by the deltas
	// Python: np.matmul(X.T, ...)
	xr, xc := X.Dims()
	res := mat.NewDense(xc, 1, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X.T(), deltas)

	// Apply "2 * and / X.shape[0]" by scaling by: 2 / nrows
	// Python: 2 * ... / X.shape[0]
	res.Scale(2.0/float64(xr), res)
	return res
}

