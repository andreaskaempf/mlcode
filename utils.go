// Utility functions, mainly for testing


package main

import (
	"math"
	"gonum.org/v1/gonum/mat"
)

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
func matSame(A, B mat.Matrix) bool {
    ar, ac := A.Dims()
    br, bc := B.Dims()
    if ar != br || ac != bc {
        return false
    }
    for i := 0; i < ar; i++ {
        for j := 0; j < ac; j++ {
            if ! close(A.At(i, j), B.At(i, j)) {
                return false
            }
        }
    }
    return true
}

