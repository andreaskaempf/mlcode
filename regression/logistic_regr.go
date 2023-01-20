// Logistic Regression
//
// Based on Python implementation in Chapter 5 of "Programming Machine
// Learning" by Paolo Perrotta
//
// Sample usage:
//	data, _ := readMatrixCSV("data/police.txt")
//	X := extractCols(data, 0, 2) // all cols except last
//	Y := extractCols(data, 3, 3) // just the last col
//  m := LogisticRegression{} // create model object
//  m.verbose = true  // set an attribute
//	m.train(X, Y) // train the model
//	matPrint(m.w)  // show resulting coefficients

package regression

import (
	"fmt"
	"math"
	"mlcode/utils"

	"gonum.org/v1/gonum/mat"
)

// Structure for a logistic regression model
type LogisticRegression struct {
	lr         float64    // learning rate, default .001
	tol        float64    // tolerance to stop training, default .001
	iterations int        // max iterations, default 1000
	verbose    bool       // messages during train, default false
	w          *mat.Dense // vector of weights, set during training
}

// Train a logistic regression model
func (m *LogisticRegression) Train(X, Y *mat.Dense) {

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

	// Just repeat for given number of interations
	for i := 0; i < m.iterations; i++ {

		// Calculate loss
		l := m.Loss(X, Y)
		if m.verbose {
			fmt.Printf("Iteration %d: loss = %f\n", i, l)
		}

		// Adjust the weights (coefficients) using the gradients
		// Python: w -= gradient(X, Y, w) * lr
		grads := m.Gradient(X, Y)
		grads.Scale(m.lr, grads)
		m.w.Sub(m.w, grads)
	}

}

// Forward prediction given X values and weights (coefficients)
// Python: sigmoid(np.matmul(X, w))
func (m *LogisticRegression) Forward(X *mat.Dense) *mat.Dense {

	// weighted_sum = np.matmul(X, w)
	xr, _ := X.Dims()
	res := mat.NewDense(xr, 1, nil)
	res.Mul(X, m.w)

	// return sigmoid(weighted_sum) -- must be vectorized
	res.Apply(func(i, j int, v float64) float64 {
		return utils.Sigmoid(v)
	}, res)
	return res
}

// Predict (classify) given matrix of features and vector of weights
// (coefficients)
// Python: np.round(forward(X, w))
func (m *LogisticRegression) Classify(X *mat.Dense) *mat.Dense {
	preds := m.Forward(X)
	preds.Apply(func(i, j int, v float64) float64 {
		return math.Round(v)
	}, preds)
	return preds
}

// Calculate loss function for predictions vs. actual values
func (m *LogisticRegression) Loss(X, Y *mat.Dense) float64 {

	// Calculate predictions
	// Python: y_hat = forward(X, w)
	// where X: 30x3, Y: 30x1, w: 3x1
	y_hat := m.Forward(X) // 30x1

	// Calculate average loss, using direct calculation rather than operations on matrices.
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
func (m *LogisticRegression) Gradient(X, Y *mat.Dense) *mat.Dense {

	// Get differences of predictions vs. actual
	// Python: (forward(X, w) - Y))
	deltas := m.Forward(X)
	deltas.Sub(deltas, Y)

	// Multiply transposed X by the deltas
	// Python: np.matmul(X.T, ...)
	xr, xc := X.Dims()
	res := mat.NewDense(xc, 1, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X.T(), deltas)

	// Apply "2 * and / X.shape[0]" by scaling by: 2 / nrows
	// Python: 2 * ... / X.shape[0]
	//res.Scale(2.0/float64(xr), res)
	res.Scale(1.0/float64(xr), res) // No, don't multiply by 2
	return res
}

// Function to demonstrate logistic regression
func Logistic_Regression_Demo() {

	// Read data into matrix, separate X and Y
	df, _ := utils.ReadCSV("data/police.txt")
	data := df.ToMatrix()

	// Separate X and Y
	X := utils.ExtractCols(data, 0, 2) // all cols except last
	Y := utils.ExtractCols(data, 3, 3) // just the last col
	fmt.Println("X =")
	utils.MatPrint(X)
	fmt.Println("Y =")
	utils.MatPrint(Y)

	// Train logistic regression model
	m := LogisticRegression{verbose: true}
	m.Train(X, Y)
	fmt.Println("\nFinal coefficients:")
	utils.MatPrint(m.w)
}
