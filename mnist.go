// Functions to read MNIST digits data.  The data files can be
// downloaded from http://yann.lecun.com/exdb/mnist/
// and should be stored in data/mnist under the current directory,
// in their original gzipped state.
//
// AK, 27/09/2022

package mlcode

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Demonstration of logistic regression on MNIST digits data, achieves 95%
// accuracy on identifying the digit 5 (but 10/90 imbalanced data set).
func mnist_demo() {

	// Read training images and labels
	pics := ReadImages("data/mnist/train-images-idx3-ubyte.gz")
	labs := ReadLabels("data/mnist/train-labels-idx1-ubyte.gz")

	// Read test data
	tpics := ReadImages("data/mnist/t10k-images-idx3-ubyte.gz")
	tlabs := ReadLabels("data/mnist/t10k-labels-idx1-ubyte.gz")

	// Training labels: for multi-class, one-hot-encode the digits, so there is
	// one row per label, and 10 columns, one for each possible digit 0-9
	labs1 := OneHotEncode(labs) // not required for test labels

	// For binary classifier, convert labels to 1 if 5 or 0 otherwise
	// labs[i] = ifThenElse(labs[i] == 5, 1, 0)
	/*labs1 := mat.NewDense(len(labs), 1, nil)
	for i := 0; i < len(labs); i++ {
		if labs[i] == 5 {
			labs1.Set(i, 0, 1)
		}
	}*/

	// Create and train binary logistic regression model,
	//m := LogisticRegression{iterations: 100, lr: .00001, verbose: true}
	//m.train(pics1, labs1)

	// Train multi-class classifier using 10-column one-hot encoded labels
	m := MultiLogRegression{iterations: 100, lr: 1e-5, verbose: true}
	m.Train(pics, labs1)

	// Predict on test data, measure simple accuracy
	fmt.Println("Predicting")
	preds := m.Classify(tpics) // should training pics include bias?
	var ok, n int
	nlabs, _ := tlabs.Dims()
	for i := 0; i < nlabs; i++ {
		n += 1
		if tlabs.At(i, 0) == preds.At(i, 0) {
			ok += 1
		}
	}
	fmt.Printf("Accuracy = %f\n", float64(ok)/float64(n))
}

// Read MNIST images file, returns a matrix with one image per row,
// bias value of 1 inserted at the beginning of each row
func ReadImages(filename string) *mat.Dense {

	// Open raw binary file
	f, err := os.Open(filename)
	if err != nil {
		panic("Could not open file")
	}
	defer f.Close()

	// Prepare to read as gzip
	fz, err := gzip.NewReader(f)
	if err != nil {
		panic("Could not open as gzip file")
	}
	defer fz.Close()

	// Read header: magic 2051, #images, rows, cols
	header := make([]int32, 4, 4)
	err = binary.Read(fz, binary.BigEndian, header)
	if err != nil {
		panic("Could not read binary header")
	}
	n := int(header[1])
	nrows := header[2]
	ncols := header[3]
	npix := int(nrows * ncols) // pixels per image
	fmt.Printf("%s: %d images, %d x %d\n", filename, n, nrows, ncols)

	// Allocate matrix for result, one image per row, with bias number
	// at beginning of each row
	result := mat.NewDense(n, npix+1, nil)

	// Read each image
	pixels := make([]uint8, npix, npix) // buffer for one image
	for i := 0; i < n; i++ {            // each image

		// Read 28x28 pixels
		err := binary.Read(fz, binary.BigEndian, pixels)
		if err != nil {
			panic("Could not read image")
		}

		// Convert to floats, inserting bias at beginning
		pixels2 := make([]float64, npix+1, npix+1)
		pixels2[0] = 1 // bias
		for j := 0; j < npix; j++ {
			pixels2[j+1] = float64(pixels[j])
		}

		// Set row of the result
		result.SetRow(i, pixels2)
	}

	return result
}

// Read MNIST labels file, returns N x 1 matrix
func ReadLabels(filename string) *mat.Dense {

	// Open raw binary file (TODO: gzip)
	f, err := os.Open(filename)
	if err != nil {
		panic("Could not open file")
	}
	defer f.Close()

	// Prepare to read as gzip
	fz, err := gzip.NewReader(f)
	if err != nil {
		panic("Could not open as gzip file")
	}
	defer fz.Close()

	// Read header: magic 2049, #images
	header := make([]int32, 2, 2)
	err = binary.Read(fz, binary.BigEndian, header)
	if err != nil {
		panic("Could not read binary")
	}
	n := int(header[1])
	fmt.Printf("%s: %d labels\n", filename, n)

	// Allocate matrix for result, one label per row
	result := mat.NewDense(n, 1, nil)

	// Read all labels, assign to matrix
	labels := make([]uint8, n, n)
	err = binary.Read(fz, binary.BigEndian, labels)
	if err != nil {
		panic("Could not read label")
	}
	for i := 0; i < n; i++ { // each label
		result.Set(i, 0, float64(labels[i]))
	}

	return result
}

// One-hot encode a column of 0-9 labels
// TODO: don't hardcode the number of values
func OneHotEncode(labels *mat.Dense) *mat.Dense {
	nr, _ := labels.Dims() // TODO: check nc is 1
	nvals := 10            // TODO: infer from data
	result := mat.NewDense(nr, nvals, nil)
	for i := 0; i < nr; i++ {
		result.Set(i, int(labels.At(i, 0)), 1)
	}
	return result
}
