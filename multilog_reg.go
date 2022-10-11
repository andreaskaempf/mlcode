// Multi-class Logistic Regression
//
// Based on Python implementation in Chapter 7 of "Programming Machine
// Learning" by Paolo Perrotta

package mlcode

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Structure for a multi-class logistic regression model
type MultiLogRegression struct {
	lr         float64    // learning rate, e.g., .001
	iterations int        // iterations to run
	verbose    bool       // messages during train, default false
	w          *mat.Dense // vector of weights, set during training
}

// Train a multi-class logistic regression model, sets the weights in the model object
func (m *MultiLogRegression) Train(X, Y *mat.Dense) {

	// Initialize weights/coefficients to zero
	// Python: np.zeros((X_train.shape[1], Y_train.shape[1]))
	_, xc := X.Dims()
	_, yc := Y.Dims()
	m.w = mat.NewDense(xc, yc, nil)

	// Just repeat for given number of interations
	for i := 0; i < m.iterations; i++ {

		// Calculate loss (only for information purposes)
		if m.verbose {
			l := m.Loss(X, Y)
			fmt.Printf("Iteration %d: loss = %f\n", i, l)
		}

		// Adjust the weights (coefficients) using the gradients
		// Python: w -= gradient(X, Y, w) * lr
		grads := m.Gradient(X, Y)
		grads.Scale(m.lr, grads)
		m.w.Sub(m.w, grads)
		//matPrint(m.w)
	}

}

// Compute the gradient for logistic regression
// Python: return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
func (m *MultiLogRegression) Gradient(X, Y *mat.Dense) *mat.Dense {

	// Get differences of predictions vs. actual
	// Python: (forward(X, w) - Y))
	deltas := m.Forward(X)
	deltas.Sub(deltas, Y)

	// Multiply transposed X by the deltas
	// Python: np.matmul(X.T, ...)
	_, dc := deltas.Dims()
	xr, xc := X.Dims()
	res := mat.NewDense(xc, dc, nil) // TODO: Can we avoid allocating each time?
	res.Mul(X.T(), deltas)

	// Divide by number of rows: ... / X.shape[0]
	res.Scale(1.0/float64(xr), res)
	return res
}

// Forward prediction given X values and weights (coefficients)
// Python: sigmoid(np.matmul(X, w))
func (m *MultiLogRegression) Forward(X *mat.Dense) *mat.Dense {

	// weighted_sum = np.matmul(X, w)
	xr, _ := X.Dims()
	_, wc := m.w.Dims()
	res := mat.NewDense(xr, wc, nil) // TODO: Avoid allocating each time?
	res.Mul(X, m.w)

	// return sigmoid(weighted_sum) -- must be vectorized
	res.Apply(func(i, j int, v float64) float64 {
		return Sigmoid(v)
	}, res)

	return res
}

// Calculate loss function for predictions vs. actual values
func (m *MultiLogRegression) Loss(X, Y *mat.Dense) float64 {

	// Calculate predictions
	// Python: y_hat = forward(X, w)
	y_hat := m.Forward(X)

	// Calculate average loss, using direct calculation rather than
	// operations on matrices.
	// Python:
	//   first_term = Y * np.log(y_hat)
	//   second_term = (1 - Y) * np.log(1 - y_hat)
	//   return -np.sum(first_term + second_term) / X.shape[0]
	yrows, ycols := Y.Dims()
	xrows, _ := X.Dims()
	var result float64
	for i := 0; i < yrows; i++ {
		for j := 0; j < ycols; j++ {
			result += Y.At(i, j) * math.Log(y_hat.At(i, j))
			result += (1 - Y.At(i, j)) * math.Log(1-y_hat.At(i, j))
		}
	}

	return result / float64(xrows) * -1
}

// Predict (classify) given matrix of features and vector of weights
// (coefficients)
// Python: labels = np.argmax(y_hat, axis=1)
//         return labels.reshape(-1, 1)
func (m *MultiLogRegression) Classify(X *mat.Dense) *mat.Dense {

	// Just predict forward, and return a vector of the column numbers
	// with the highest value
	preds := m.Forward(X)
	rows, _ := preds.Dims()
	result := mat.NewDense(rows, 1, nil) // TODO: Avoid allocating each time?
	for r := 0; r < rows; r++ {
		result.Set(r, 0, float64(MaxCol(preds, r)))
	}
	return result
}
