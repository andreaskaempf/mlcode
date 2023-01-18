// Random forest, using decision tree

package decision_tree

import (
	"fmt"
	"math/rand"
	"mlcode/dataframe"
	"mlcode/utils"
)

// A random forest is just a list of trained decision trees
type Forest []Node

// Demonstrate random forest, using Titanic data set
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
	nTrees := 200
	fmt.Println("Training", nTrees, "decision trees")
	forest := RandomForest(df, "Survived", nTrees)

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

// Create/train a random forest
func RandomForest(df *dataframe.DataFrame, depv string, nTrees int) *Forest {
	trees := Forest{} //[]*Node{}
	for i := 0; i < nTrees; i++ {
		sample := SampleWithReplacement(df)
		tree := DecisionTree(sample, depv, 0)
		trees = append(trees, *tree)
	}
	return &trees
}

// Predict with a random forest
func RandomForestPredict(forest *Forest, row *dataframe.DataFrame) string {

	// Make a prediction with each tree
	preds := []string{}
	for _, tree := range *forest {
		pred1 := Predict(&tree, row)
		preds = append(preds, pred1)
	}

	// Use the most common prediction as the prediction for the model
	return utils.MostCommon(preds)
}

// Sample a dataframe with replacement, resulting in same number of rows
func SampleWithReplacement(df *dataframe.DataFrame) *dataframe.DataFrame {

	// Start with an empty dataframe, same structure
	df2 := df.CopyStructure()

	// Keep sampling random rows until the new dataframe is same size
	nrows := df.NRows()
	for n := 0; n < nrows; n++ {
		i := rand.Intn(nrows)
		df2.CopyRow(df, i)
	}

	// Return the new dataframe
	return df2
}
