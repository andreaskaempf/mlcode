// gonum_utils.go
//
// Utility functions for matrices using gonum

package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// Pretty-print a matrix
func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// Extract columns from a matrix and return a copy. First and last
// columns are zero-based and inclusive, so 0, 1 indicates the first
// two columns.
// Would prefer to use m.Slice() but that returns a mat.Matrix which
// cannot be passed as mat.Dense argument.
func extractCols(m *mat.Dense, i, j int) *mat.Dense {

	// Make sure values are valid. Arguments represent the starting
	// and ending column numbers, both zero based and inclusive (so 0,0 is
	// the first column)
	rows, cols := m.Dims()
	if j == -1 { // allow -1 to indicate the last column
		j = cols - 1
	}
	if i < 0 || i >= cols || j < 0 || j >= cols || j < i {
		panic("Invalid call to extractCols")
	}

	// Create a slice with the values to be extracted.
	// Slice(i, r, j, c) returns matrix view with i,j at top left,
	// and up to row r and col c. So Slice(0,5,0,2) returns
	// top-left slice of 5 rows & 2 cols.
	// Slice arguments are row numbers, starting at zero.
	// Starting coords (i, j) are zero-based row/column numbers to start at,
	// and are inclusive. Ending coordinates (torow, tocol) are zero based
	// row/column numbers but are not inclusive.
	vals := m.Slice(0, rows, i, j+1)

	// Allocate and return a new copy
	return mat.DenseCopyOf(vals)
}

// Read a matrix from a CSV file, assumes first row is column headings
// and all values are numeric (i.e., no string values)
func readMatrixCSV(filename string) (*mat.Dense, []string) {

	// Read CSV file, will return empty array if failed
	data := readCsv(filename)
	if len(data) < 2 { // must be at least headings plus one row of data
		return nil, []string{}
	}

	// Parse numbers from remaining rows into a vector of floats
	headings := data[0]    // First row is headings
	nrows := len(data) - 1 // number of rows, excluding headers
	ncols := len(headings) // number of columns
	nums := make([]float64, nrows*ncols)
	i := 0 // current cell
	for r := 1; r <= nrows; r++ {
		row := data[r]
		if len(row) != ncols {
			fmt.Printf("Row %d: %d instead of %d columns, skipping\n", r+1, len(row), ncols)
			continue // should probably quit instead of skipping row
		}
		for c := 0; c < ncols; c++ {
			s := row[c]
			n, err := strconv.ParseFloat(s, 64)
			if err != nil {
				fmt.Printf("Row %d: invalid number \"%s\", replacing with 0\n", r+1, s)
				n = 0
			}
			nums[i] = n
			i++
		}
	}

	// Convert to a gonum matrix
	m := mat.NewDense(nrows, ncols, nums)
	return m, headings
}

// Read CSV file into list of lists of strings,
// return empty array if file not found or can't parse
func readCsv(filename string) [][]string {

	// Open file
	f, err := os.Open(filename)
	if err != nil {
		return [][]string{}
	}
	defer f.Close()

	// Read all rows into list of string lists
	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		return [][]string{}
	}

	return data
}

// Insert a column of 1s in front of a matrix
func PrependBias(m *mat.Dense) *mat.Dense {

	// Make a new matrix, with additional column
	r, c := m.Dims()
	m1 := mat.NewDense(r, c+1, nil)

	// Copy values across, shifted over one column, and set first col to 1
	for j := 0; j < r; j++ { // each row
		m1.Set(j, 0, 1.0)        // set first col to 1
		for i := 0; i < c; i++ { // each column
			m1.Set(j, i+1, m.At(j, i)) // copy value, shift right one
		}
	}

	return m1
}

// Return the numbers of a matrix as a list of floats, row-wise
func MatrixNums(m *mat.Dense) []float64 {
	r, c := m.Dims()
	a := make([]float64, r*c, r*c)
	i := 0                      // current index in output array
	for ri := 0; ri < r; ri++ { // each row of matrix
		for ci := 0; ci < c; ci++ { // each column
			a[i] = m.At(ri, ci)
			i++
		}
	}
	return a
}

// Write matrix to a text file for debugging
func writeMatrix(m *mat.Dense, filename string) {
	nr, nc := m.Dims()
	fmt.Printf("Writing matrix %d x %d to %s\n", nr, nc, filename)
	f, _ := os.Create(filename)
	for r := 0; r < nr; r++ {
		f.WriteString("[")
		for c := 0; c < nc; c++ {
			if c > 0 {
				f.WriteString(", ")
			}
			fmt.Fprintf(f, "%d", int(m.At(r, c)))
		}
		f.WriteString("]\n")
	}
	f.Close()
	fmt.Println("  finished")
}
