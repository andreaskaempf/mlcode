// Demo of Support Vector Machine Model, using breast cancer data set

package svm

import (
	"fmt"
	"mlcode/utils"

	"gonum.org/v1/gonum/mat"
)

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
