// Two demos of the decision tree

package decision_tree

import (
	"fmt"
	"mlcode/utils"
)

// Demo of the decision tree classifier, using the Iris dataset
// (text labels, all other variables floating point)
func DecisionTreeDemo() {

	// Read Iris data set from CSV file
	df, err := utils.ReadCSV("data/iris.csv")
	if err != nil {
		panic(err)
	}

	// Create a decision tree
	tree := DecisionTree(df, "variety", 0)
	PrintTree(tree, 0)

	// Make predictions
	correct := 0
	for i := 0; i < df.NRows(); i++ {
		row := df.GetRow(i)
		pred := Predict(tree, row)
		actual := row.GetColumn("variety").Strings[0]
		if pred == actual {
			correct++
		}
	}

	// Report accuracy
	acc := float64(correct) / float64(df.NRows()) * 100
	fmt.Println(correct, "of", df.NRows(), "correct =", acc, "%")
}

// Another demo, using the Titanic data set (predict 1/0, other variables
// are mix of text, integer, and floating point)
func DecisionTreeDemo2() {

	// Read Titanic data set from CSV file
	df := GetTitanicData("data/titanic.csv")
	if !df.Check() {
		return
	}

	// Parameters for training the decision trees
	MaxDepth = 5
	MinLeaf = 1

	// Create a decision tree to predict survival
	tree := DecisionTree(df, "Survived", 0)
	PrintTree(tree, 0)

	if !df.Check() {
		return
	}

	// Make predictions
	correct := 0
	for i := 0; i < df.NRows(); i++ {
		row := df.GetRow(i)
		pred := Predict(tree, row)
		actual := row.GetColumn("Survived").Strings[0]
		if pred == actual {
			correct++
		}
	}

	// Report accuracy
	acc := float64(correct) / float64(df.NRows()) * 100
	fmt.Println(correct, "of", df.NRows(), "correct =", acc, "%")
}
