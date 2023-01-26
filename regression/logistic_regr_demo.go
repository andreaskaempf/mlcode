// Function to demonstrate logistic regression

package regression

import (
	"fmt"
	"mlcode/utils"
)

func Logistic_Regression_Demo() {

	// Read data into matrix, separate X and Y
	df, _ := utils.ReadCSV("data/police.txt")
	data := df.ToMatrix()

	// Separate X and Y
	X := utils.ExtractCols(data, 0, 2) // all cols except last
	Y := utils.ExtractCols(data, 3, 3) // just the last col
	fmt.Println("X =")
	utils.MatPrint(X)
	fmt.Println("Y =")
	utils.MatPrint(Y)

	// Train logistic regression model
	m := LogisticRegression{verbose: true}
	m.Train(X, Y)
	fmt.Println("\nFinal coefficients:")
	utils.MatPrint(m.w)
}
