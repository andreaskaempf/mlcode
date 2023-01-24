// https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
// https://www.mltut.com/svm-implementation-in-python-from-scratch/
// https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f
// https://www.kaggle.com/code/prathameshbhalekar/svm-with-kernel-trick-from-scratch

package svm

import (
	"fmt"
	"math"
	"mlcode/utils"

	"gonum.org/v1/gonum/mat"
)

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

	// Normalize X values, between zero and one (TODO: Normalize Y?)
	for i := 0; i < len(*feats); i++ {
		normalize(&(*feats)[i].Floats)
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
	//utils.MatPrint(X)
	//utils.MatPrint(Y)

	// Train the  model
	// W = sgd(X_train.to_numpy(), y_train.to_numpy())
	W := sgd(X, Y)
	utils.MatPrint(W)
}

// Global hyperparameters
var maxEpochs int = 5000
var regularizationStrength float64 = 10000
var learningRate float64 = 0.000001
var costThreshold float64 = 0.01 // stop when improvement less than this fraction

// Train stochastic gradient descent
func sgd(X, Y *mat.Dense) *mat.Dense {

	// Initialize weights as zeros
	nr, nc := X.Dims()
	W := mat.NewVecDense(nc, nil)

	// Iterate until no more improvement, or maximum iterations
	prevCost := math.MaxFloat64
	for epoch := 1; epoch <= maxEpochs; epoch++ {

		// TODO: X, Y = shuffle(features, outputs)

		// Do each row
		for i := 0; i < nr; i++ {

			x := X.RowView(i) // mat.Vector
			y := Y.At(i, 0)   // float64
			ascent := calculateCostGradient(W, x, y)

			// W = W - (learningRate * ascent)
			ascent.Scale(learningRate, ascent)
			W.Sub(W, ascent)
		}

		// Stop when converged
		cost := computeCost(W, X, Y)
		fmt.Printf("Epoch %d: cost = %f\n", epoch, cost)
		if math.Abs(prevCost-cost) < costThreshold*prevCost {
			break
		}
		prevCost = cost

		// DEBUG: Stop after one iteration
		break
	}

	// Return final weights
	return W
}

// Compute cost gradient for training SVM
func calculateCostGradient(W *mat.Dense, x mat.Vector, y float64) *mat.Dense {

	// distance = 1 - (Y_batch * np.dot(X_batch, W))
	_, nc := x.Dims() // x and W both 31 x 1
	dist := mat.NewDense(1, nc, nil)
	dist.Mul(x.T(), W)
	dist.Scale(y, dist)
	dist.Apply(func(i, j int, v float64) float64 {
		return 1 - v
	}, dist)

	// dw = np.zeros(len(W))
	wr, _ := W.Dims()
	dw := mat.NewDense(wr, 1, nil) // 31 x 1

	//for ind, d in enumerate(distance):
	//    if max(0, d) == 0:  // i.e., d < 0
	//        di = W
	//    else:
	//        di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
	//    dw += di

	//dw = dw/len(Y_batch)  # average
	return dw
}

// Compute cost for SVM
// W (30,) X (455, 30) Y (455,) distances (455,)
func computeCost(W, X, Y *mat.Dense) float64 {

	//showDims("W", W) // 1 x 569
	//showDims("X", X) // 569 x 31
	//showDims("Y", Y) // 569 x 1

	// Calculate distances
	// Python: distances = 1 - Y * np.dot(X, W)
	//    np.dot(X, W) => (455,)
	//    Y * "        => (455,)
	//    1 - "        => (455,)
	// distances[distances < 0] = 0  # i.e., max(0, distance)

	nr, nc := X.Dims() // 569, 31
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

	// sumDistances= np.sum(distances)
	var sumDist float64
	//showDims("dist", dist)
	for i := 0; i < nr; i++ {
		sumDist += dist.At(i, 0)
	}

	// Calculate hinge loss
	// N = X.shape[0]
	// hinge_loss = regularization_strength * (sumDistances / N)
	hingeLoss := regularizationStrength * (sumDist / float64(nr))

	// Calculate cost
	// cost = 1 / 2 * np.dot(W, W) + hinge_loss
	var cost float64 = 0
	//showDims("W", W)
	for c := 0; c < nc; c++ {
		w := W.At(c, 0)
		cost += w * w
	}
	cost += hingeLoss
	return cost / 2
}

func showDims(name string, X *mat.Dense) {
	nr, nc := X.Dims()
	fmt.Printf("%s: %d x %d\n", name, nr, nc)
}
func showDimsV(name string, X mat.Vector) {
	nr, nc := X.Dims()
	fmt.Printf("%s: %d x %d\n", name, nr, nc)
}

// Normalize a list of floats between 0 and 1, in-place
func normalize(nums *[]float64) {
	mn := min(*nums)
	mx := max(*nums)
	for i := 0; i < len(*nums); i++ {
		(*nums)[i] = ((*nums)[i] - mn) / (mx - mn)
	}
}

// Minimum of a list
func min[T int | int64 | float64 | string](list []T) T {
	res := list[0]
	for i := 1; i < len(list); i++ {
		if list[i] < res {
			res = list[i]
		}
	}
	return res
}

// Maximum of a list
func max[T int | int64 | float64 | string](list []T) T {
	res := list[0]
	for i := 1; i < len(list); i++ {
		if list[i] > res {
			res = list[i]
		}
	}
	return res
}
