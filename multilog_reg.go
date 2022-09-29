// Multi-class Logistic Regression
//
// Based on Python implementation in Chapter 7 of "Programming Machine
// Learning" by Paolo Perrotta

package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Structure for a logistic regression model
type MultiLogRegression struct {
	lr         float64    // learning rate, default .001
	tol        float64    // tolerance to stop training, default .001
	iterations int        // max iterations, default 1000
	verbose    bool       // messages during train, default false
	w          *mat.Dense // vector of weights, set during training
}

// Train a multi-class logistic regression model
func (m *MultiLogRegression) train(X, Y *mat.Dense) {

	// Initialize weights/coefficients to zero (a column vector, with length =
	// number of columns in X)
	_, xc := X.Dims()
	_, yc := Y.Dims()
	m.w = mat.NewDense(xc, yc, nil) // Python: np.zeros((X_train.shape[1], Y_train.shape[1]))

	// Set other model parameters if not set yet
	if m.iterations <= 0 {
		m.iterations = 1000
	}
	if m.lr <= 0 || m.lr >= 1 {
		m.lr = .001
	}
	if m.tol <= 0 || m.tol >= 1 {
		m.tol = .001
	}

	// Just repeat for given number of interations
	for i := 0; i < m.iterations; i++ {

		// Calculate loss
		l := m.loss(X, Y)
		if m.verbose {
			fmt.Printf("Iteration %d: loss = %f\n", i, l)
		}

		// Adjust the weights (coefficients) using the gradients
		// Python: w -= gradient(X, Y, w) * lr
		grads := m.gradient(X, Y)
		grads.Scale(m.lr, grads)
		m.w.Sub(m.w, grads)
	}

}

// Forward prediction given X values and weights (coefficients)
// Python: sigmoid(np.matmul(X, w))
func (m *MultiLogRegression) forward(X *mat.Dense) *mat.Dense {

	// weighted_sum = np.matmul(X, w)
	xr, _ := X.Dims()
	_, wc := m.w.Dims()
	res := mat.NewDense(xr, wc, nil)
	res.Mul(X, m.w)

	// return sigmoid(weighted_sum) -- must be vectorized
	res.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, res)

	return res
}

// Predict (classify) given matrix of features and vector of weights
// (coefficients)
// Python: labels = np.argmax(y_hat, axis=1)
//         return labels.reshape(-1, 1)
func (m *MultiLogRegression) classify(X *mat.Dense) *mat.Dense {

	// Just predict forward, and return a vector of the column numbers with the highest value
	preds := m.forward(X)
	rows, _ := preds.Dims()
	result := mat.NewDense(rows, 1, nil)
	for r := 0; r < rows; r++ {
		result.Set(r, 0, float64(maxCol(preds, r)))
	}
	return result
}

// Return the column that has the maximum value in the given row
func maxCol(m *mat.Dense, row int) int {
	_, cols := m.Dims()
	maxVal := m.At(row, 0)
	maxCol := 0
	for i := 1; i < cols; i++ {
		if m.At(row, i) > maxVal {
			maxVal = m.At(row, i)
			maxCol = i
		}
	}
	return maxCol
}

// Calculate loss function for predictions vs. actual values
func (m *MultiLogRegression) loss(X, Y *mat.Dense) float64 {

	// Calculate predictions
	// Python: y_hat = forward(X, w)
	// where X: 30x3, Y: 30x1, w: 3x1
	y_hat := m.forward(X) // 30x1

	// Calculate average loss, using direct calculation rather than operations on matrices.
	// Python:
	//   first_term = Y * np.log(y_hat)
	//   second_term = (1 - Y) * np.log(1 - y_hat)
	//   return -np.average(first_term + second_term)
	// Where: Y 30x1, y_hat 30x1
	yrows, _ := Y.Dims()
	xrows, _ := X.Dims()
	var result float64
	for i := 0; i < yrows; i++ {
		result += Y.At(i, 0)*math.Log(y_hat.At(i, 0)) + (1-Y.At(i, 0))*math.Log(1-y_hat.At(i, 0))
	}
	return result / float64(xrows) * -1
}

// Compute the gradient for logistic regression
// Python: return 2 * np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
func (m *MultiLogRegression) gradient(X, Y *mat.Dense) *mat.Dense {

	// Get differences of predictions vs. actual
	// Python: (forward(X, w) - Y))
	deltas := m.forward(X)
	deltas.Sub(deltas, Y)

	// Multiply transposed X by the deltas
	// Python: np.matmul(X.T, ...)
	_, dc := deltas.Dims()
	xr, xc := X.Dims()
	res := mat.NewDense(xc, dc, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X.T(), deltas)

	// Apply "2 * and / X.shape[0]" by scaling by: 2 / nrows
	// Python: 2 * ... / X.shape[0]
	res.Scale(1.0/float64(xr), res)
	return res
}
