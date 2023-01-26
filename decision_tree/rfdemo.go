// Demonstrate random forest, using Titanic data set

package decision_tree

import (
	"fmt"
	"mlcode/utils"
)

func RandomForestDemo() {

	// Read Titanic data set from CSV file
	df := GetTitanicData("data/titanic.csv")
	if !df.Check() {
		return
	}

	// Parameters for training the decision trees
	MaxDepth = 5
	MinLeaf = 1

	// Train lots of trees, sampling data with replacement
	nTrees := 2000
	fmt.Println("Training", nTrees, "decision trees")
	forest := RandomForest2(df, "Survived", nTrees)

	// Make predictions, by predicting for each tree, then using most common value
	fmt.Println("Making predictions")
	correct := 0
	for i := 0; i < df.NRows(); i++ {

		// Get this row in the original data set
		row := df.GetRow(i)

		// Predict using random forest
		pred := RandomForestPredict(forest, row)

		// Compare to actuals
		actual := row.GetColumn("Survived").Strings[0]
		if pred == actual {
			correct++
		}
	}

	// Report accuracy
	acc := float64(correct) / float64(df.NRows()) * 100
	fmt.Println(correct, "of", df.NRows(), "correct =", acc, "%")
}

// Read and prepare the Titanic data set
func GetTitanicData(filename string) *utils.DataFrame {

	// Read Titanic data set from CSV file
	utils.MISSING_INT = -1
	utils.MISSING_FLOAT = -1.0
	df, err := utils.ReadCSV(filename)
	if err != nil {
		panic(err)
	}

	// Remove some columns we don't need for the model
	// TODO: Drop rows with missing values
	df = df.DropColumns([]string{"PassengerId", "Name", "Ticket"})

	// Turn "Survived" column into a string
	surv := df.GetColumn("Survived")
	utils.Assert(surv.Dtype == "int64" && len(surv.Ints) > 0 && len(surv.Strings) == 0, "Survived malformed")
	for _, si := range surv.Ints {
		yesNo := utils.IfThenElse(si == 1, "Yes", "No")
		surv.Strings = append(surv.Strings, yesNo)
	}
	surv.Ints = nil
	surv.Dtype = "string"

	// Replace Cabin with just the first letter (would indicate class?)
	cabin := df.GetColumn("Cabin")
	for i, c := range cabin.Strings {
		if len(c) > 1 {
			cabin.Strings[i] = c[:1]
		}
	}

	// Return the data set
	return df
}
