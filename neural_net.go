// Simple 3-layer neural network, based on chapters 9-12 of "Programming
// Machine Learning" by Paolo Perotta

package mlcode

import (
	//"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Forward propagation, through a single hidden layer
// Python:
//   def forward(X, w1, w2):
//     h = sigmoid(np.matmul(prepend_bias(X), w1))
//     y_hat = softmax(np.matmul(prepend_bias(h), w2))
//     return y_hat
func Forward(X, w1, w2 *mat.Dense) *mat.Dense {

	// Make a copy of X with bias prepended
	X1 := PrependBias(X)

	// Multiply X1 by w1
	h1 := mat.NewDense(MatRows(X1), MatCols(w1), nil)
	h1.Mul(X1, w1)

	// Prepend bias to the new hidden layer
	h1a := PrependBias(h1)

	// Multiply hidden by w2
	y := mat.NewDense(MatRows(h1a), MatCols(w2), nil)
	y.Mul(h1a, w2)

	return y
}

// Loss function
// Python:
//   def loss(Y, y_hat):
//     return -np.sum(Y * np.log(y_hat)) / Y.shape[0]
func Loss(Y, yHat *mat.Dense) float64 {

	// Take logs of y_hat
	nr, nc := yHat.Dims()
	logs := mat.NewDense(nr, nc, nil)
	logs.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, logs)

	// Element-wise multiply Y by y-hat logs
	logs.MulElem(Y, logs)

	// Sum the result and divide by number of rows
	var loss float64
	for r := 0; r < nr; r++ {
		for c := 0; c < nc; c++ {
			loss += logs.At(r, c)
		}
	}
	return loss / float64(nr)
}

// Classify, taking highest probability for each instance
// Python:
//   def classify(X, w1, w2):
//     y_hat = forward(X, w1, w2)
//     labels = np.argmax(y_hat, axis=1)
//     return labels.reshape(-1, 1)
// Predict (classify) given matrix of features and vector of weights
// (coefficients)
// Python: labels = np.argmax(y_hat, axis=1)
//         return labels.reshape(-1, 1)
func Classify(X, w1, w2 *mat.Dense) *mat.Dense {

	// Just predict forward, and return a vector of the column numbers
	// with the highest value
	// TODO: Doesn't this need to deal with more than one dimension?
	preds := Forward(X, w1, w2)
	nr, _ := preds.Dims()
	result := mat.NewDense(nr, 1, nil) // TODO: Avoid allocating each time?
	for r := 0; r < nr; r++ {
		result.Set(r, 0, float64(MaxCol(preds, r)))
	}
	return result
}

// The softmax function, on a matrix, each row adds up to 1
// Python:
// def softmax(logits):
//     exponentials = np.exp(logits)
//     return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)
// Sample input:
//   0.3, 0.8, 0.2
//   0.1, 0.9, 0.1
// Produces output:
//   0.2814, 0.4640, 0.2546
//   0.2367, 0.5267, 0.2367
func SoftMax(m *mat.Dense) *mat.Dense {

	// Create a matrix of the exponentials of the inputs
	nr, nc := m.Dims()
	exps := mat.NewDense(nr, nc, nil) // TODO: avoid reallocating each time?
	exps.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, m)

	// For each row, get the total, then divide each value by the row total
	for r := 0; r < nr; r++ {
		var rt float64 // row total
		for c := 0; c < nc; c++ {
			rt += exps.At(r, c)
		}
		for c := 0; c < nc; c++ {
			exps.Set(r, c, exps.At(r, c)/rt) // TODO: check for div by zero?
		}
	}
	return exps
}
