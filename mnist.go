// Logistic regression on MNIST digits data, achieves 95% accuracy on
// identifying the digit 5 (but 10/90 imbalanced data set).
//
// The data files can be downloaded from http://yann.lecun.com/exdb/mnist/
// and should be stored in data/mnist under the current directory, in their
// original gzipped state.
//
// AK, 27/09/2022

package main

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Structure for header of images file
type ImageHeader struct {
	Signature  uint32 // seems to be 2051, not used
	N          uint32 // number of images
	Cols, Rows uint32 // rows & cols for each image
}

// Structure for header of labels file
type LabelHeader struct {
	Signature uint32 // seems to be 2051, not used
	N         uint32 // number of labels
}

// Demonstrate training logistic regression binary classification
// on one digit in MNIST data
func main() {

	// Read training images and labels
	pics := readMNISTIMages("data/mnist/train-images-idx3-ubyte.gz")
	labs := readMNISTLabels("data/mnist/train-labels-idx1-ubyte.gz")
	fmt.Printf("%d images, %d labels read\n", len(pics), len(labs))
	if len(pics) == 0 || len(labs) == 0 || len(pics) != len(labs) {
		panic("Bad data")
	}

	// Add a column of 1s to each image, for bias
	for i := 0; i < len(pics); i++ {
		pics[i] = *PrependBias(&pics[i]) // TODO: avoid pointer magic?
	}

	// Convert images to a single matrix, one flattened image per row
	// WARNING: the bias gets repeated for every row, is this okay?
	ir, ic := pics[0].Dims()
	pics1 := mat.NewDense(len(pics), ir*ic, nil)
	for r := 0; r < len(pics); r++ { // copy each image a row
		nums := MatrixNums(&pics[r])
		pics1.SetRow(r, nums)
	}

	// For multi-class, one-hot-encode the digits, so there is one row per
	// label, and 10 columns, one for each possible digit 0-9
	labsMulti := mat.NewDense(len(labs), 10, nil)
	for i := 0; i < len(labs); i++ {
		labsMulti.Set(i, int(labs[i]), 1)
	}

	// For binary classifier, convert labels to 1 if 5 or 0 otherwise
	// labs[i] = ifThenElse(labs[i] == 5, 1, 0)
	labs1 := mat.NewDense(len(labs), 1, nil)
	for i := 0; i < len(labs); i++ {
		if labs[i] == 5 {
			labs1.Set(i, 0, 1)
		}
	}

	// Create binary logistic regression model, set parameters
	//m := LogisticRegression{}
	//m.iterations = 100 // 1000 iterations gets to loss .211
	//m.lr = .00001      // need .00002 or smaller to converge
	//m.verbose = true   // to show running loss value

	// Train binary classifier logistic regression model on image data with 1/0 labels
	//m.train(pics1, labs1)

	// Train multi-class classifier using 10-column one-hot encoded labels
	m := MultiLogRegression{iterations: 20000, lr: .000001, verbose: true}
	m.train(pics1, labsMulti)

	//fmt.Println("\nFinal coefficients:")
	//matPrint(m.w)

	// Predict
	preds := m.classify(pics1)

	// Measure simple accuracy
	var ok, n int
	for i := 0; i < len(pics); i++ {
		n += 1
		if labsMulti.At(i, 0) == preds.At(i, 0) {
			ok += 1
		}
	}
	fmt.Printf("Accuracy = %f\n", float64(ok)/float64(n))
}

// Read MNIST images file, returns a slice of 28x28 matrices
func readMNISTIMages(filename string) []mat.Dense {

	// Images go into an array
	pics := []mat.Dense{}

	// Open raw binary file
	f, err := os.Open(filename)
	if err != nil {
		fmt.Println("Could not open file:", filename)
		return pics
	}
	defer f.Close()

	// Prepare to read as gzip
	fz, err := gzip.NewReader(f)
	if err != nil {
		fmt.Println("Could not open as gzip file:", filename)
		return pics
	}
	defer fz.Close()

	// Read header from first 16 bytes
	bb := make([]byte, 16)
	_, err = fz.Read(bb)
	if err != nil {
		fmt.Println("Could not read header")
		return pics
	}

	// Coerce to structure
	h := ImageHeader{}
	buf := bytes.NewBuffer(bb)
	err = binary.Read(buf, binary.BigEndian, &h)
	if err != nil {
		fmt.Println("Could not parse binary")
		return pics
	}

	fmt.Printf("Header for %s: %+v\n", filename, h)

	// Read images into matrix format
	for i := 0; i < int(h.N); i++ {

		// Read the image data
		pixels := make([]byte, h.Rows*h.Cols)
		_, err = fz.Read(pixels)
		if err != nil {
			fmt.Println("Could not read image", i)
			return pics // warning: list will be incomplete
		}

		// Convert to a matrix, add to array
		p := mat.NewDense(int(h.Rows), int(h.Cols), bytesToFloats(pixels))
		pics = append(pics, *p)
	}

	// Return list of images
	return pics
}

// Read MNIST labels file, returns a slice of floats
func readMNISTLabels(filename string) []float64 {

	// Images go into an array
	labs := []float64{}

	// Open raw binary file
	f, err := os.Open(filename)
	if err != nil {
		fmt.Println("Could not open file:", filename)
		return labs
	}
	defer f.Close()

	// Prepare to read as gzip
	fz, err := gzip.NewReader(f)
	if err != nil {
		fmt.Println("Could not open as gzip file:", filename)
		return labs
	}
	defer fz.Close()

	// Read header from first 8 bytes
	bb := make([]byte, 8)
	_, err = fz.Read(bb)
	if err != nil {
		fmt.Println("Could not read header")
		return labs
	}

	// Coerce to structure
	h := LabelHeader{}
	buf := bytes.NewBuffer(bb)
	err = binary.Read(buf, binary.BigEndian, &h)
	if err != nil {
		fmt.Println("Could not parse binary")
		return labs
	}

	fmt.Printf("Header for %s: %+v\n", filename, h)

	// Read labels into an array of bytes
	ll := make([]byte, h.N)
	_, err = fz.Read(ll)
	if err != nil {
		fmt.Println("Could not read labels")
		return labs
	}

	// Return list of floats
	// TODO: Matrix?
	return bytesToFloats(ll)
}
