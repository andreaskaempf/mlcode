// Utility functions, mainly for testing

package mlcode

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// The sigmoid function
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Test whether two numbers are close
func close(a, b float64) bool {
	if a == 0 && b == 0 {
		return true
	}
	return math.Abs(a-b)/((a+b)/2.0) < .001
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

// Determine if two matrices are the same (or at least close)
// TODO: You can actually use build-in function from mat for this
func matSame(A, B mat.Matrix) bool {
	ar, ac := A.Dims()
	br, bc := B.Dims()
	if ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if !close(A.At(i, j), B.At(i, j)) {
				return false
			}
		}
	}
	return true
}

// Simple if-then-else operator, like a?b:c in C,
// return a if condition is true, otherwise returns b
func ifThenElse(cond bool, a, b float64) float64 {
	if cond {
		return a
	} else {
		return b
	}
}
