// Demo of the linear regression functions.

package regression

import (
	"fmt"
	"mlcode/utils"
)

// Test/demo linear regression
func Linear_Regression_Demo() {

	// Read data
	df, _ := utils.ReadCSV("data/pizza_3_vars.txt")
	data := df.ToMatrix()
	utils.MatPrint(data)

	// Split into X columns & Y column
	X := utils.ExtractCols(data, 0, 2) // all cols except last
	Y := utils.ExtractCols(data, 3, 3) // just the last col

	// Create and train model
	m := LinearRegression{}
	m.verbose = true
	m.Train(X, Y)

	// Show coefficients
	fmt.Println("Final weights")
	utils.MatPrint(m.w) // prints final coefficients

	// Make prediction
	preds := m.Predict(X)
	fmt.Println("Predictions:")
	utils.MatPrint(preds)
}
