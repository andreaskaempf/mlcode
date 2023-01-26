// Simple classification decision tree

package decision_tree

import (
	"fmt"
	"mlcode/utils"
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

// Parameters for learning, default values may be changed
// before training tree
var MaxDepth = 3    // Maximum depth for a tree
var MinLeaf = 20    // Minimum size of a leaf
var Verbose = false // whether to show progress messages

// Create decision tree, recursively, returns top-level node
func DecisionTree(df *utils.DataFrame, depv string, level int) *Node {

	// Terminate with a leaf node it:
	// 1. too few rows left
	// 2. tree too deep
	// 3. no more variation
	dcol := df.GetColumn(depv)
	if df.NRows() < MinLeaf || level >= MaxDepth || giniIndex(dcol.Strings) == 0 {
		return &Node{Value: utils.MostCommon(dcol.Strings)}
	}

	// Define all possible splits, based on each attribute
	// and find the one that produces the lowest Gini Index
	var bestCol string       // best column to split on
	var bestSplitNum float64 // value if numeric split
	var bestSplitCat string  // value if categorical split
	var bestGini float64 = 1
	var bestLeft, bestRight *utils.DataFrame
	for _, c := range *df {

		// Skip the dependent variable
		if c.Name == depv {
			continue
		}

		// If the column is numeric, test splits at midpints between
		// all values
		if c.Dtype == "float64" || c.Dtype == "int64" {
			var splits []float64
			if c.Dtype == "float64" {
				splits = midPoints(c.Floats)
			} else if c.Dtype == "int64" {
				splits = midPoints(c.Ints)
			}
			for _, split := range splits {
				left, right := splitNumeric(*df, c.Name, split)
				leftLabels := left.GetColumn(depv).Strings
				rightLabels := right.GetColumn(depv).Strings
				if len(leftLabels) == 0 || len(rightLabels) == 0 {
					continue
				}
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
				if len(leftLabels) == 0 || len(rightLabels) == 0 {
					continue
				}
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
		} else {
			fmt.Println("Warning: column ignored, type", c.Dtype)
		}
	}

	// If there was no meaningful split found, return a terminal node
	if len(bestCol) == 0 {
		return &Node{Value: utils.MostCommon(dcol.Strings)}
	}

	// Show the best split found
	if Verbose {
		fmt.Printf("Depth %2d: n = %d, best split on %s at ", level, df.NRows(), bestCol)
		if len(bestSplitCat) > 0 {
			fmt.Printf("\"%s\"", bestSplitCat)
		} else {
			fmt.Print(bestSplitNum)
		}
		fmt.Println(" => Gini", bestGini)
	}

	// Using the best split found, recursively do left and right sides
	n := Node{SplitVar: bestCol, SplitNum: bestSplitNum, SplitCat: bestSplitCat, G: bestGini}
	n.Left = DecisionTree(bestLeft, depv, level+1)
	n.Right = DecisionTree(bestRight, depv, level+1)
	return &n
}

// Split a dataframe on a numeric (float) column, into two dataframes,
// at the given split value
func splitNumeric(df utils.DataFrame, colName string, split float64) (*utils.DataFrame, *utils.DataFrame) {

	// Make two empty dataframes with same columns
	left := df.CopyStructure()
	right := df.CopyStructure()

	// Copy rows to the appropriate dataframe, based on splits
	splitCol := df.GetColumn(colName) // the column to split on
	for i := 0; i < df.NRows(); i++ {

		// Determine if this row is less than cut-off, depends on type
		var splitLeft bool
		if splitCol.Dtype == "float64" {
			splitLeft = splitCol.Floats[i] < split
		} else if splitCol.Dtype == "int64" {
			splitLeft = float64(splitCol.Ints[i]) < split
		} else {
			panic("splitNumeric: invalid data type " + splitCol.Dtype)
		}

		// Copy to left or right
		if splitLeft {
			left.CopyRow(&df, i)
		} else {
			right.CopyRow(&df, i)
		}
	}

	// Return the left and right splits
	return left, right
}

// Split a dataframe on a categorical (string) column, into two dataframes,
// left for equal, right for not equal
func splitCategorical(df utils.DataFrame, colName string, split string) (*utils.DataFrame, *utils.DataFrame) {

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
func Predict(tree *Node, row *utils.DataFrame) string {

	// Terminal node is the prediction
	if len(tree.Value) > 0 {
		return tree.Value
	}

	// Otherwise evaluate the split
	col := row.GetColumn(tree.SplitVar)
	var predLeft bool
	if col.Dtype == "string" {
		val := col.Strings[0]
		predLeft = val == tree.SplitCat
	} else if col.Dtype == "float64" {
		val := col.Floats[0]
		predLeft = val < tree.SplitNum
	} else if col.Dtype == "int64" {
		val := float64(col.Ints[0])
		predLeft = val < tree.SplitNum
	} else {
		fmt.Println("TODO: Skipping prediction on", col.Dtype)
		return "error"
	}

	// Proceed to left or right branch
	if predLeft {
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

// For a list of numbers (integer or float), return a list that is the
// midpoints between each consecutive pair; result is always list of floats,
// even if you pass it a list of ints, since mid-points need to be floats.
func midPoints[T float64 | int64](nums []T) []float64 {
	nums = utils.Unique(nums) // remove duplicates, sorted
	res := []float64{}
	for i := 1; i < len(nums); i++ {
		mid := float64(nums[i-1]) + float64(nums[i]-nums[i-1])/2.0
		res = append(res, mid)
	}
	return res
}
