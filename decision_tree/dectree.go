// Simple classification decision tree

package decision_tree

import (
	"fmt"
	"mlcode/dataframe"
	"mlcode/utils"
	"sort"
)

// Node in a decision tree
type Node struct {
	SplitVar    string  // column name
	SplitVal    float64 // number to split at
	G           float64 // gini index at this point
	Left, Right *Node   // left and right nodes for decision
	Value       string  // terminal value if a leaf
}

// Parameters for learning (TODO: don't hard-code)
const MaxDepth = 3 // Maximum depth for a tree
const MinLeaf = 20 // Minimum size of a leaf

// Demo of the decision tree classifier, using the Iris dataset
// (text labels, all other variables floating point)
func DecisionTreeDemo() {

	// Read Iris data set from CSV file
	df, err := dataframe.ReadCSV("data/iris.csv")
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

	// Read data set from CSV file
	df, err := dataframe.ReadCSV("data/titanic.csv")
	if err != nil {
		panic(err)
	}

	// Remove some columns we don't need for the model
	df = df.DropColumns([]string{"PassengerId", "Name"})

	// Create a decision tree
	tree := DecisionTree(df, "Survived", 0)
	PrintTree(tree, 0)

	// Make predictions
	correct := 0
	for i := 0; i < df.NRows(); i++ {
		row := df.GetRow(i)
		pred := Predict(tree, row)
		actual := row.GetColumn("Survived").Ints[0]
		if pred == actual {
			correct++
		}
	}

	// Report accuracy
	acc := float64(correct) / float64(df.NRows()) * 100
	fmt.Println(correct, "of", df.NRows(), "correct =", acc, "%")
}

// Create decision tree, recursively, returns top-level node
func DecisionTree(df *dataframe.DataFrame, depv string, level int) *Node {

	// Terminate with a leaf node it:
	// 1. too few rows left
	// 2. tree too deep
	// 3. no more variation
	dcol := df.GetColumn(depv)
	if df.NRows() < MinLeaf || level >= MaxDepth || giniIndex(dcol.Strings) == 0 {
		return &Node{Value: mostCommon(dcol.Strings)}
	}

	// Define all possible splits, based on each attribute
	// and find the one that produces the lowest Gini Index
	var bestCol string
	var bestSplit float64
	var bestGini float64 = 1
	var bestLeft, bestRight *dataframe.DataFrame
	for _, c := range *df {

		// Skip the dependent variable
		if c.Name == depv {
			continue
		}

		// If the column is numeric, test splits at midpints between
		// all values (TODO: ints, strings)
		if c.Dtype == "float64" {
			splits := midPoints(c.Floats)
			for _, split := range splits {
				left, right := splitNumeric(*df, c.Name, split)
				leftLabels := left.GetColumn(depv).Strings
				rightLabels := right.GetColumn(depv).Strings
				G := giniCombined(leftLabels, rightLabels)
				if G < bestGini {
					bestGini = G
					bestCol = c.Name
					bestSplit = split
					bestLeft = left
					bestRight = right
				}
			}
		}
	}

	// Using the best split found, recursively do left and right sides
	fmt.Println(level, ": n =", df.NRows(), ", best split on", bestCol,
		"at", bestSplit, "=> Gini", bestGini)
	n := Node{SplitVar: bestCol, SplitVal: bestSplit, G: bestGini}
	n.Left = DecisionTree(bestLeft, depv, level+1)
	n.Right = DecisionTree(bestRight, depv, level+1)
	return &n
}

// Split a dataframe on a numeric (float) column, into two dataframes,
// at the given split value
func splitNumeric(df dataframe.DataFrame, colName string, split float64) (*dataframe.DataFrame, *dataframe.DataFrame) {

	// Make two empty dataframes with same columns
	left := df.CopyStructure()
	right := df.CopyStructure()

	// Copy rows to the appropriate dataframe, based on splits
	splitCol := df.GetColumn(colName) // the column to split on
	for i := 0; i < df.NRows(); i++ {
		if splitCol.Floats[i] < split {
			left.CopyRow(&df, i)
		} else {
			right.CopyRow(&df, i)
		}
	}

	return left, right
}

// Predict from a decision tree, return predicted label
func Predict(tree *Node, row *dataframe.DataFrame) string {

	// Terminal node is the prediction
	if len(tree.Value) > 0 {
		return tree.Value
	}

	// Otherwise evaluate the split
	// TODO: ints and strings
	val := row.GetColumn(tree.SplitVar).Floats[0]
	if val < tree.SplitVal {
		return Predict(tree.Left, row)
	} else {
		return Predict(tree.Right, row)
	}
}

// Print decision tree
func PrintTree(tree *Node, level int) {
	for i := 0; i < level; i++ {
		fmt.Print("  ")
	}
	if len(tree.Value) > 0 {
		fmt.Println("-->", tree.Value)
	} else {
		fmt.Println(tree.SplitVar, "<", tree.SplitVal)
		PrintTree(tree.Left, level+1)
		PrintTree(tree.Right, level+1)
	}
}

// Calculate the weighted average Gini Index for two lists of labels
func giniCombined(left, right []string) float64 {
	gl := giniIndex(left)
	gr := giniIndex(right)
	nl := float64(len(left))
	nr := float64(len(right))
	return gl*nl/(nl+nr) + gr*nr/(nl+nr)
}

// Calculate the Gini Index of one list of labels
func giniIndex(labels []string) float64 {

	// Get the count for each label
	classCount := map[string]int{}
	for _, l := range labels {
		classCount[l]++
	}

	// Calculate the Gini index
	var gini float64
	n := float64(len(labels))
	for _, c := range classCount { // each label found
		p := float64(c) / n
		gini += p * (1 - p)
	}
	return gini
}

// For a list of numbers, return a list that is the midpoints
// between each consecutive pair
func midPoints(nums []float64) []float64 {
	nums = utils.Unique(nums) // remove duplicates
	sort.Float64s(nums)       // sort ascending
	res := []float64{}
	for i := 1; i < len(nums); i++ {
		mid := nums[i-1] + (nums[i]-nums[i-1])/2
		res = append(res, mid)
	}
	return res
}

// Find the most common value in a list of strings
func mostCommon(ss []string) string {
	counts := map[string]int{}
	var highestCount int
	var mostFreq string
	for _, s := range ss {
		counts[s]++
		if counts[s] > highestCount {
			highestCount = counts[s]
			mostFreq = s
		}
	}
	return mostFreq
}
