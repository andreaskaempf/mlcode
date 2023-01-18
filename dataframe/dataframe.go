// dataframe.go
//
// Very simple data frame implementation

package dataframe

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"mlcode/utils"
	"os"
	"strconv"
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
	for {

		// Read next line of CSV file, stop if end of file
		l, err := r.Read()
		if err != nil {
			break
		}

		// If first line, populate the data frame
		if len(df) == 0 { // no columns yet
			for _, c := range l {
				col := Series{Name: c, Dtype: "string"} // assume string for now
				df = append(df, col)
			}
		} else {
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
		MISSING_INT := int64(math.MaxInt64)
		MISSING_FLOAT := math.MaxFloat64
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
		if !utils.In(s.Name, names) {
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
func (dest *DataFrame) CopyRow(src *DataFrame, row int) {
	for c := 0; c < len(*src); c++ {
		if (*src)[c].Dtype == "string" {
			(*dest)[c].Strings = append((*dest)[c].Strings, (*src)[c].Strings[row])
		} else if (*src)[c].Dtype == "int64" {
			(*dest)[c].Ints = append((*dest)[c].Ints, (*src)[c].Ints[row])
		} else if (*src)[c].Dtype == "float64" {
			(*dest)[c].Floats = append((*dest)[c].Floats, (*src)[c].Floats[row])
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
			src.Ints = append(col.Ints, src.Ints[row])
		} else if (*df)[c].Dtype == "float64" {
			col.Floats = append(col.Floats, src.Floats[row])
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
