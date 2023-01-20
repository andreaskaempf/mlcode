// dataframe.go
//
// Very simple data frame implementation

package utils

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// A data frame is just a list of columns
type DataFrame []Series

// One column of a data frame
type Series struct {
	Name    string
	Dtype   string // int64, float64 or string
	Ints    []int64
	Floats  []float64
	Strings []string
}

// Values to use for missing
var MISSING_INT int64 = math.MaxInt64
var MISSING_FLOAT float64 = math.MaxFloat64

// Read CSV file into a data frame, with columns as integers, float64, or strings,
// missing values set to maximum in or float
func ReadCSV(filename string) (*DataFrame, error) {

	// Open file
	f, err := os.Open(filename)
	if err != nil {
		return nil, errors.New("File not found: " + filename)
	}

	// Read CSV file, populate columns as strings for now
	df := DataFrame{} // new, empty data frame
	r := csv.NewReader(f)
	line_no := 0
	for {

		// Read next line of CSV file, stop if end of file
		l, err := r.Read()
		if err != nil {
			break
		}
		line_no++

		// If first line, populate the data frame
		if len(df) == 0 { // no columns yet
			for _, c := range l {
				col := Series{Name: c, Dtype: "string"} // assume string for now
				df = append(df, col)
			}
		} else {
			if len(l) != len(df) {
				fmt.Println("WARNING: row %d has %d instead of %d columns, ignored\n", line_no, len(l), len(df))
			}
			for i, c := range l {
				df[i].Strings = append(df[i].Strings, c)
			}
		}
	}

	// Check each column in case floats or ints
	for i := range df { // each column index

		// Try converting each value to int and float
		allInts := true
		allFloats := true
		ints := []int64{}
		floats := []float64{}
		for _, c := range df[i].Strings {

			// Try int
			if allInts {
				if len(c) == 0 {
					ints = append(ints, MISSING_INT)
				} else {
					i, err := strconv.ParseInt(c, 10, 64)
					if err == nil {
						ints = append(ints, i)
					} else {
						allInts = false
					}
				}
			}

			// Try float
			if allFloats {
				if len(c) == 0 {
					floats = append(floats, MISSING_FLOAT)
				} else {
					f, err := strconv.ParseFloat(c, 64)
					if err == nil {
						floats = append(floats, f)
					} else {
						allFloats = false
					}
				}
			}
		}

		// If all integers, cast column as integer, otherwise if all floats,
		// cast column as float, otherwise leave as string
		if allInts {
			df[i].Dtype = "int64"
			df[i].Ints = ints
			df[i].Strings = nil
		} else if allFloats {
			df[i].Dtype = "float64"
			df[i].Floats = floats
			df[i].Strings = nil
		}
	}

	return &df, nil
}

// Number of rows in a dataframe
func (df *DataFrame) NRows() int {
	c0 := (*df)[0] // use the first column
	if c0.Dtype == "string" {
		return len(c0.Strings)
	} else if c0.Dtype == "int64" {
		return len(c0.Ints)
	} else if c0.Dtype == "float64" {
		return len(c0.Floats)
	} else {
		panic("nrows: dataframe has invalid data type")
	}
}

// Get one column from a dataframe
func (df *DataFrame) GetColumn(name string) *Series {
	for i := 0; i < len(*df); i++ {
		s := &(*df)[i]
		if s.Name == name {
			return s
		}
	}
	return nil
}

// Drop one or more columns from a dataframe (returns new copy)
func (df *DataFrame) DropColumns(names []string) *DataFrame {
	df2 := DataFrame{}
	for _, s := range *df {
		if !In(s.Name, names) {
			df2 = append(df2, s)
		}
	}
	return &df2
}

// Create an empty dataframe with same structure as an existing one
func (df *DataFrame) CopyStructure() *DataFrame {
	df2 := DataFrame{}
	for _, col := range *df {
		df2 = append(df2, Series{Name: col.Name, Dtype: col.Dtype})
	}
	return &df2
}

// Copy a row from one dataframe to another, assumes both
// have the same structure (i.e., column types)
// TODO: Can we avoid all the pointers?
func (dest *DataFrame) CopyRow(src *DataFrame, row int) {
	for c := 0; c < len(*src); c++ {
		if (*src)[c].Dtype == "string" {
			(*dest)[c].Strings = append((*dest)[c].Strings, (*src)[c].Strings[row])
		} else if (*src)[c].Dtype == "int64" {
			(*dest)[c].Ints = append((*dest)[c].Ints, (*src)[c].Ints[row])
		} else if (*src)[c].Dtype == "float64" {
			(*dest)[c].Floats = append((*dest)[c].Floats, (*src)[c].Floats[row])
		} else {
			panic("CopyRow: invalid data type " + (*src)[c].Dtype)
		}
	}
}

// Extract one row from a dataframe, returns a dataframe of one row
func (df *DataFrame) GetRow(row int) *DataFrame {
	df2 := DataFrame{}
	for c := 0; c < len(*df); c++ {
		src := (*df)[c]
		col := Series{Name: src.Name, Dtype: src.Dtype}
		if src.Dtype == "string" {
			col.Strings = append(col.Strings, src.Strings[row])
		} else if src.Dtype == "int64" {
			col.Ints = append(col.Ints, src.Ints[row])
		} else if src.Dtype == "float64" {
			col.Floats = append(col.Floats, src.Floats[row])
		} else {
			panic("GetRow: invalid column type")
		}
		df2 = append(df2, col)
	}
	return &df2
}

// Show summary, i.e., number of rows, column descriptors
func (df *DataFrame) Summary() {
	fmt.Printf("Dataframe with %d rows, %d cols:\n", df.NRows(), len(*df))
	for _, c := range *df {
		fmt.Println(" ", c.Name, c.Dtype)
	}
}

// Check a dataframe for validity
func (df *DataFrame) Check() bool {

	// Check each column for type, invalid data
	nrows := df.NRows()
	ok := true
	for _, c := range *df {
		ns := len(c.Strings)
		ni := len(c.Ints)
		nf := len(c.Floats)
		var nd int
		var invalid bool
		if c.Dtype == "string" {
			invalid = nf > 0 || ni > 0
			nd = ns
		} else if c.Dtype == "float64" {
			invalid = ns > 0 || ni > 0
			nd = nf
		} else if c.Dtype == "int64" {
			invalid = ns > 0 || nf > 0
			nd = ni
		} else {
			fmt.Println("WARNING: Column", c.Name, "has invalid type", c.Dtype)
			ok = false
		}
		if invalid {
			fmt.Println("WARNING:", c.Dtype, "column", c.Name, "has values of another type")
			ok = false
		}
		if nd != nrows {
			fmt.Println("WARNING: column", c.Name, "has invalid number of rows")
			ok = false
		}
	}
	return ok
}

// Convert numeric columns of a dataframe to a Gonum matrix
func (df *DataFrame) ToMatrix() *mat.Dense {

	// Extract just the numeric columns, into one long list
	nrows := df.NRows()
	cols := []Series{}
	for _, c := range *df {
		if c.Dtype == "float64" || c.Dtype == "int64" {
			cols = append(cols, c)
		}
	}
	if nrows == 0 || len(cols) == 0 {
		fmt.Println("ToMatrix: dataframe has no rows or no numeric columns, cannot convert")
		return nil
	}

	// Convert values to a long list of numbers, one row at a time
	n := nrows * len(cols)        // number of cells
	nums := make([]float64, n, n) // pre-allocate array
	i := 0
	for ri := 0; ri < nrows; ri++ {
		for _, c := range cols {
			if c.Dtype == "float64" {
				nums[i] = c.Floats[ri]
			} else {
				nums[i] = float64(c.Ints[ri])
			}
			i++
		}
	}

	// Convert to matrix
	return mat.NewDense(nrows, len(cols), nums)
}
