// Utility functions, mainly for testing

package utils

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// The sigmoid function
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Test whether two numbers are close
func Close(a, b float64) bool {
	if a == 0 && b == 0 {
		return true
	}
	return math.Abs(a-b)/((a+b)/2.0) < .001
}

// Test whether two lists of floats are the same
func Same(A, B []float64) bool {
	if len(A) != len(B) {
		return false
	}
	for i := 0; i < len(A); i++ {
		if !Close(A[i], B[i]) {
			return false
		}
	}
	return true
}

// Determine if two matrices are the same (or at least close)
// TODO: You can actually use build-in function from mat for this
func MatSame(A, B mat.Matrix) bool {
	ar, ac := A.Dims()
	br, bc := B.Dims()
	if ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if !Close(A.At(i, j), B.At(i, j)) {
				return false
			}
		}
	}
	return true
}

// Simple if-then-else operator, like a?b:c in C,
// return a if condition is true, otherwise returns b
func IfThenElse[T int | int64 | float64 | byte | string](cond bool, a, b T) T {
	if cond {
		return a
	} else {
		return b
	}
}

// Is element in a list?
func In[T int | int64 | float64 | byte | string](c T, s []T) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return true
		}
	}
	return false
}

// Unique values in a list, sorted
func Unique[T int | int64 | float64 | byte | string](values []T) []T {

	// Collect values found into a map
	vals := map[T]int{}
	for _, v := range values {
		vals[v] = 1
	}

	// Turn the map into a list
	result := []T{}
	for v, _ := range vals {
		result = append(result, v)
	}

	// Sort the list
	sort.Slice(result, func(i, j int) bool {
		return result[i] < result[j]
	})
	return result
}

// Find the most common value in a list
func MostCommon[T int | int64 | float64 | byte | string](ss []T) T {
	counts := map[T]int{}
	var highestCount int
	var mostFreq T
	for _, s := range ss {
		counts[s]++
		if counts[s] > highestCount {
			highestCount = counts[s]
			mostFreq = s
		}
	}
	return mostFreq
}

// Determine if all elements in a list are the same
func allSame[T int | int64 | float64 | byte | string](labels []T) bool {
	classCount := map[T]int{}
	for _, l := range labels {
		classCount[l]++
	}
	return len(classCount) == 1
}

// Panic if a test condition is not true
func Assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}
