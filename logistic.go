// Logistic Regression
//
// Based on Python implementation in Chapter 5 of "Programming Machine
// Learning" by Paolo Perrotta

package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)


// Function to demonstrate logistic regression
func logistic_regression_demo() {

	// Read data, convert to matrix
	m, _ := readMatrixCSV("data/police.txt")
	//fmt.Println("Headings:", h)
	//fmt.Println("Data =")
	matPrint(m)

	// Separate matrices for X and Y
	X := extractCols(m, 0, 2) // all cols except last
	Y := extractCols(m, 3, 3) // just the last col
	fmt.Println("X =")
	matPrint(X)
	fmt.Println("Y =")
	matPrint(Y)

	// Train linear regression model
	w := trainLogRegr(X, Y, .001, 1000, true)
	fmt.Println("\nFinal coefficients:")
	matPrint(w)

}

func trainLogRegr(X, Y *mat.Dense, lr float64, iters int, verbose bool) *mat.Dense {

	// Initialize weights/coefficients to zero (a column vector, with length =
	// number of columns in X)
	_, c := X.Dims()
	w := mat.NewDense(c, 1, nil) // Python: np.zeros((X.shape[1], 1))

	// Just repeat for given number of interations
	for i := 0; i < iters; i++ {

		l := lossLogRegr(X, Y, w)
		if verbose {
			fmt.Printf("Iteration %d: loss = %f\n", i, l)
		}

		// Adjust the weights (coefficients) using the gradients
		// Python: w -= gradient(X, Y, w) * lr
		grads := gradientLogReg(X, Y, w)

		grads.Scale(lr, grads)
		w.Sub(w, grads)
	}

	return w
}

// The sigmoid function
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Forward prediction given X values and weights (coefficients)
// Python: sigmoid(np.matmul(X, w))
func forward(X, w *mat.Dense) *mat.Dense {

	// weighted_sum = np.matmul(X, w)
	xr, _ := X.Dims()
	res := mat.NewDense(xr, 1, nil)
	res.Mul(X, w)

	// return sigmoid(weighted_sum) -- must be vectorized
	res.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, res)
	return res
}

// Predict (classify) given matrix of features and vector of weights
// (coefficients)
// Python: np.round(forward(X, w))
func classify(X, w *mat.Dense) *mat.Dense {
	preds := forward(X, w)
	preds.Apply(func(i, j int, v float64) float64 {
		return math.Round(v)
	}, preds)
	return preds
}

// Calculate loss function for predictions vs. actual values, for logistic
// regression
func lossLogRegr(X, Y, w *mat.Dense) float64 {

	// Calculate predictions
	// Python: y_hat = forward(X, w)
	// where X: 30x3, Y: 30x1, w: 3x1
	y_hat := forward(X, w) // 30x1

	// Calculate average loss, using direct calculation
	// rather than operations on matrices.
	// Python:
	//   first_term = Y * np.log(y_hat)
	//   second_term = (1 - Y) * np.log(1 - y_hat)
	//   return -np.average(first_term + second_term)
	// Where: Y 30x1, y_hat 30x1
	rows, _ := Y.Dims()
	var result float64
	for i := 0; i < rows; i++ {
		result += Y.At(i, 0)*math.Log(y_hat.At(i, 0)) + (1-Y.At(i, 0))*math.Log(1-y_hat.At(i, 0))
	}
	return result / float64(rows) * -1
}

// Compute the gradient for logistic regression
// Python: return 2 * np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
func gradientLogReg(X, Y, w *mat.Dense) *mat.Dense {

	// Get differences of predictions vs. actual
	// Python: (forward(X, w) - Y))
	deltas := forward(X, w)
	deltas.Sub(deltas, Y)

	// Multiply transposed X by the deltas
	// Python: np.matmul(X.T, ...)
	xr, xc := X.Dims()
	res := mat.NewDense(xc, 1, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X.T(), deltas)

	// Apply "2 * and / X.shape[0]" by scaling by: 2 / nrows
	// Python: 2 * ... / X.shape[0]
	//res.Scale(2.0/float64(xr), res)
	res.Scale(1.0/float64(xr), res)  // No, don't multiply by 2
	return res
}
