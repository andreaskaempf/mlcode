// Random forest, using decision tree

package decision_tree

import (
	"math/rand"
	"mlcode/utils"
)

// A random forest is just a list of trained decision trees
type Forest []Node

// Create/train a random forest, single threaded
func RandomForest(df *utils.DataFrame, depv string, nTrees int) *Forest {
	trees := Forest{}
	for i := 0; i < nTrees; i++ {
		sample := SampleWithReplacement(df)
		tree := DecisionTree(sample, depv, 0)
		trees = append(trees, *tree)
	}
	return &trees
}

// Create/train a random forest, with concurrency
func RandomForest2(df *utils.DataFrame, depv string, nTrees int) *Forest {

	// Initialize an empty forest
	trees := Forest{}

	// Create channel
	ch := make(chan *Node)

	// Launch all the trees in background
	for i := 0; i < nTrees; i++ {
		go createTree(df, depv, ch)
	}

	// Collect all the trees into a list
	for i := 0; i < nTrees; i++ {
		t := <-ch
		trees = append(trees, *t)
	}

	// Return list of trees
	return &trees
}

// Create a tree, for concurrent random forest creation
func createTree(df *utils.DataFrame, depv string, ch chan *Node) {
	sample := SampleWithReplacement(df)
	tree := DecisionTree(sample, depv, 0)
	ch <- tree
}

// Predict with a random forest
func RandomForestPredict(forest *Forest, row *utils.DataFrame) string {

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
func SampleWithReplacement(df *utils.DataFrame) *utils.DataFrame {

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
