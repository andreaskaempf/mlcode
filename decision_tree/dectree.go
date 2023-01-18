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
	SplitNum    float64 // number to split at
	SplitCat    string  // or string to split on
	G           float64 // gini index at this point
	Left, Right *Node   // left and right nodes for decision
	Value       string  // terminal value if a leaf
}

// Parameters for learning (TODO: don't hard-code)
var MaxDepth = 3 // Maximum depth for a tree
var MinLeaf = 20 // Minimum size of a leaf

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
	dataframe.MISSING_INT = -1
	dataframe.MISSING_FLOAT = -1.0
	df, err := dataframe.ReadCSV("data/titanic.csv")
	if err != nil {
		panic(err)
	}

	// Remove some columns we don't need for the model
	// TODO: Drop rows with missing nulls?
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

	// Create a decision tree to predict survival
	MaxDepth = 10
	MinLeaf = 5 // TODO: if this is too low, crashes on Titanic data set
	tree := DecisionTree(df, "Survived", 0)
	PrintTree(tree, 0)

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
	var bestCol string       // best column to split on
	var bestSplitNum float64 // value if numeric split
	var bestSplitCat string  // value if categorical split
	var bestGini float64 = 1
	var bestLeft, bestRight *dataframe.DataFrame
	for _, c := range *df {

		// Skip the dependent variable
		if c.Name == depv {
			continue
		}

		// If the column is numeric, test splits at midpints between
		// all values (TODO: ints)
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
					bestSplitNum = split
					bestSplitCat = ""
					bestLeft = left
					bestRight = right
				}
			}
		} else if c.Dtype == "string" { // split on categorical variable
			splits := utils.Unique(c.Strings) // all possible values
			for _, split := range splits {
				left, right := splitCategorical(*df, c.Name, split)
				leftLabels := left.GetColumn(depv).Strings
				rightLabels := right.GetColumn(depv).Strings
				G := giniCombined(leftLabels, rightLabels)
				if G < bestGini {
					bestGini = G
					bestCol = c.Name
					bestSplitNum = 0 // n/a
					bestSplitCat = split
					bestLeft = left
					bestRight = right
				}
			}
		}
	}

	// Show the best split found
	fmt.Printf("%d: n = %d, best split on %s at ", level, df.NRows(), bestCol)
	if len(bestSplitCat) > 0 {
		fmt.Printf("\"%s\"", bestSplitCat)
	} else {
		fmt.Print(bestSplitNum)
	}
	fmt.Println(" => Gini", bestGini)

	// Using the best split found, recursively do left and right sides
	n := Node{SplitVar: bestCol, SplitNum: bestSplitNum, SplitCat: bestSplitCat, G: bestGini}
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

// Split a dataframe on a categorical (string) column, into two dataframes,
// left for equal, right for not equal
func splitCategorical(df dataframe.DataFrame, colName string, split string) (*dataframe.DataFrame, *dataframe.DataFrame) {

	// Make two empty dataframes with same columns
	left := df.CopyStructure()
	right := df.CopyStructure()

	// Copy rows to the appropriate dataframe, based on splits
	splitCol := df.GetColumn(colName) // the column to split on
	for i := 0; i < df.NRows(); i++ {
		if splitCol.Strings[i] == split {
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
	// TODO: ints
	col := row.GetColumn(tree.SplitVar)
	if col.Dtype == "string" {
		val := col.Strings[0]
		if val == tree.SplitCat {
			return Predict(tree.Left, row)
		} else {
			return Predict(tree.Right, row)
		}
	} else if col.Dtype == "float64" {
		val := col.Floats[0]
		if val < tree.SplitNum {
			return Predict(tree.Left, row)
		} else {
			return Predict(tree.Right, row)
		}
	} else {
		fmt.Println("TODO: Skipping prediction on", col.Dtype)
		return "error"
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
		if len(tree.SplitCat) > 0 {
			fmt.Printf("%s == \"%s\"\n", tree.SplitVar, tree.SplitCat)
		} else {
			fmt.Printf("%s < %.2f\n", tree.SplitVar, tree.SplitNum)
		}
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
