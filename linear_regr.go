// Linear Regression
//
// Based on Python implementation in Chapters 2-4 of "Programming Machine
// Learning" by Paolo Perrotta.
//
// Sample usage (also see unit test file):
// m := LinearRegression{}   // create model
// m.verbose = true  // set flag (also lr, iterations, tol)
// m.train(X, Y)   // train the  model
// matPrint(m.w) // prints final coefficients
// preds := m.predict(X) // make prediction

package mlcode

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Structure for a linear regression model
type LinearRegression struct {
	lr         float64    // learning rate, default .001
	tol        float64    // tolerance to stop training, default .001
	iterations int        // max iterations, default 1000
	verbose    bool       // messages during train, default false
	w          *mat.Dense // vector of weights, set during training
}

// Train linear regression model using gradient descent on coeffients,
// until loss stops improving by at least the tolerance. Sets vector
// of coefficients.
func (m *LinearRegression) Train(X, Y *mat.Dense) {

	// Initialize weights/coefficients to zero (a column vector, with length =
	// number of columns in X)
	_, c := X.Dims()
	m.w = mat.NewDense(c, 1, nil) // Python: np.zeros((X.shape[1], 1))

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

	// Initialize the previous loss, so we can detect when we are converging
	prevLoss := 0.0

	// Iterate until tolerance is low
	for i := 0; i < m.iterations; i++ { // run to maximum iterations

		// Calculate loss using current weights
		l := m.Loss(X, Y)
		if m.verbose {
			fmt.Printf("Iteration %d: loss = %f\n", i, l)
		}

		// Solution found if improvement less than tolerance
		if i > 0 && math.Abs(prevLoss-l) < m.tol {
			if m.verbose {
				fmt.Println("Solution found")
			}
			return
		}

		// Remember loss for next iteration
		prevLoss = l

		// Adjust the weights (coefficients) using the gradients
		// Python: w -= gradient(X, Y, w) * lr
		grads := m.Gradient(X, Y)
		grads.Scale(m.lr, grads)
		m.w.Sub(m.w, grads)
	}

	// Message if reached max iterations
	if m.verbose {
		fmt.Printf("Stopped at %d iterations\n", m.iterations)
	}
}

// Predict Y values (one column), given X values (one column per variable) and
// coefficients (vector of values, one per X column)
// Python: return np.matmul(X, w)
// TODO: make sure weights are allocated and compatible
func (m *LinearRegression) Predict(X *mat.Dense) *mat.Dense {
	xr, _ := X.Dims()
	res := mat.NewDense(xr, 1, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X, m.w)
	return res
}

// Calculate the mean squared difference between predicted and actual values
// Python: np.average((predict(X, w) - Y) ** 2)
func (m *LinearRegression) Loss(X, Y *mat.Dense) float64 {

	// Get differences of predictions vs. actual
	deltas := m.Predict(X)
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
func (m *LinearRegression) Gradient(X, Y *mat.Dense) *mat.Dense {

	// Get differences of predictions vs. actual
	// Python: (predict(X, w) - Y))
	deltas := m.Predict(X)
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

// Test/demo linear regression
func linear_regression_demo() {
	//func main() {

	// Read data, split into X & Y
	data, _ := ReadMatrixCSV("data/pizza_3_vars.txt")
	X := ExtractCols(data, 0, 2) // all cols except last
	Y := ExtractCols(data, 3, 3) // just the last col

	// Create and train model
	//w := trainLinRegr(X, Y, .001, .001, true)
	m := LinearRegression{}
	m.verbose = true
	m.Train(X, Y)

	// Show coefficients
	fmt.Println("Final weights")
	MatPrint(m.w) // prints final coefficients

	// Make prediction
	preds := m.Predict(X)
	fmt.Println("Predictions:")
	MatPrint(preds)
}
