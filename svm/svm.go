// svm.go
//
// Support vector machine implementation, based mainly on:
// https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
//
// Also useful:
// https://www.mltut.com/svm-implementation-in-python-from-scratch/
// https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f
// https://www.kaggle.com/code/prathameshbhalekar/svm-with-kernel-trick-from-scratch
//
// AK, 26/01/2023

package svm

import (
	"fmt"
	"math"
	"mlcode/utils"

	"gonum.org/v1/gonum/mat"
)

// Global hyperparameters
var maxIterations int = 5000
var regularizationStrength float64 = 10000
var learningRate float64 = 0.000001
var costThreshold float64 = 0.01 // stop when improvement less than this fraction

// Demo of Support Vector Machine Model, using breast cancer data set
func SVMDemo() {

	// Read the breast cancer dataset
	df, err := utils.ReadCSV("data/breastcancer.csv")
	if err != nil {
		panic("Could not find data set")
	}

	// Convert "diagnosis" (the "Y" column we are trying to predict)
	// from M/B string to 1.0/-1.0
	diag := df.GetColumn("diagnosis")
	for i := 0; i < len(diag.Strings); i++ {
		f := utils.IfThenElse(diag.Strings[i] == "M", 1.0, -1.0)
		diag.Floats = append(diag.Floats, f)
	}
	diag.Dtype = "float64"
	diag.Strings = nil

	// Remove the ID and diagnosis column from the dataframe
	feats := df.DropColumns([]string{"id", "diagnosis"})

	// Normalize X values, between zero and one
	for i := 0; i < len(*feats); i++ {
		utils.Normalize(&(*feats)[i].Floats)
	}

	// Add an intercept column
	icept := utils.Series{Name: "intercept", Dtype: "float64"}
	icept.Floats = make([]float64, feats.NRows(), feats.NRows())
	for i := 0; i < feats.NRows(); i++ {
		icept.Floats[i] = 1
	}
	*feats = append(*feats, icept)

	// TODO: Remove correlated or insignificant columns

	// TODO: Split into test/train sets

	// Convert dataframes X and Y to matrices
	X := feats.ToMatrix()
	dfY := utils.DataFrame{}
	dfY = append(dfY, *diag)
	Y := dfY.ToMatrix()

	// Train the model, returns final weights
	W := sgd(X, Y)

	// Make predictions and compare accuracy
	nr, _ := X.Dims()
	preds := mat.NewVecDense(nr, nil)
	preds.MulVec(X, W)

	// Measure accuracy
	ok := 0
	for i := 0; i < nr; i++ {
		if utils.SameSign(preds.AtVec(i), Y.At(i, 0)) {
			ok++
		}
	}
	fmt.Println("Accuracy =", float64(ok)/float64(nr))
}

// Train model using stochastic gradient descent
func sgd(X, Y *mat.Dense) *mat.VecDense {

	// Initialize weights as a vector of zeros
	nr, nc := X.Dims()
	W := mat.NewVecDense(nc, nil)
	fmt.Printf("Initial cost = %f\n", computeCost(W, X, Y))

	// Iterate until no more improvement, or maximum iterations
	prevCost := math.MaxFloat64
	for iter := 1; iter <= maxIterations; iter++ {

		// Do each row, keep adjusting weights
		// TODO: X, Y = shuffle(features, outputs)
		for i := 0; i < nr; i++ {

			// Get gradient for this row
			x := X.RowView(i) // mat.Vector
			y := Y.At(i, 0)   // float64
			ascent := calculateCostGradient(W, &x, y)

			// Python: W = W - (learningRate * ascent)
			ascent.ScaleVec(learningRate, ascent)
			W.SubVec(W, ascent)
		}

		// Stop when converged, i.e., no more improvement
		cost := computeCost(W, X, Y)
		fmt.Printf("Iteration %d: cost = %f\n", iter, cost)
		if math.Abs(prevCost-cost) < costThreshold*prevCost {
			break
		}
		prevCost = cost
	}

	// Return final weights
	return W
}

// Compute cost gradient for training SVM
// Assumes X and W are vectors (one row), Y is one number
func calculateCostGradient(W *mat.VecDense, x *mat.Vector, y float64) *mat.VecDense {

	// Calculate total distance
	// Python: d = 1 - (Y * np.sum(X * W))
	// was: distance = 1 - (Y_batch * np.dot(X_batch, W))
	nc := (*x).Len()
	var dist float64
	for i := 0; i < nc; i++ { // equivalent to: Y * np.sum(X * W)
		dist += W.AtVec(i) * (*x).AtVec(i) * y
	}
	if dist < 1 {
		dist = 1 - dist
	} else {
		dist = 0
	}

	// Calculate dw
	// dw = np.zeros(len(W))
	// if dist > 0:
	//     dw = W - (regularization_strength * Y * X)
	dw := mat.NewVecDense(nc, nil) // 31 x 1
	if dist > 0 {                  // dist no longer used!
		for i := 0; i < nc; i++ {
			dw.SetVec(i, W.AtVec(i)-regularizationStrength*y*(*x).AtVec(i))
		}
	}

	// Return a vector of weight differences
	return dw
}

// Compute cost for SVM
func computeCost(W *mat.VecDense, X, Y *mat.Dense) float64 {

	// Calculate distances
	// Python: distances = 1 - Y * np.dot(X, W)
	// distances[distances < 0] = 0  # i.e., max(0, distance)
	nr, nc := X.Dims()
	dist := mat.NewDense(nr, 1, nil)
	dist.Mul(X, W)
	dist.MulElem(Y, dist)
	dist.Apply(func(i, j int, v float64) float64 {
		if v < 1 {
			return 1 - v
		} else {
			return 0
		}
	}, dist)

	// Python: sumDistances = np.sum(distances)
	var sumDist float64
	for i := 0; i < nr; i++ {
		sumDist += dist.At(i, 0)
	}

	// Calculate dot(W, W)
	var cost float64 = 0
	for c := 0; c < nc; c++ {
		w := W.AtVec(c)
		cost += w * w
	}

	// Calculate cost
	// Python: cost = 1 / 2 * np.dot(W, W) + hinge_loss
	hingeLoss := regularizationStrength * (sumDist / float64(nr))
	return cost/2 + hingeLoss
}
