// Unit tests for decision tree

package decision_tree

import (
	"math"
	"testing"
)

func TestGini(t *testing.T) {

	left := []string{"red", "red", "red", "red", "red", "blue"}
	right := []string{"blue", "blue", "blue", "blue"}

	if math.Abs(giniIndex(left)-.27778) > .0001 {
		t.Errorf("Left %f instead of .278", giniIndex(left))
	}
	if giniIndex(right) != 0 {
		t.Errorf("Left %f instead of 0", giniIndex(right))
	}
	comb := giniCombined(left, right)
	if math.Abs(comb-.16667) > .0001 {
		t.Errorf("Combined %f instead of .167", comb)
	}
}
