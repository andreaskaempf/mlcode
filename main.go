// main.go
//
// Driver for demonstrating/testing the various machine learning models

package main

import (
	"fmt"
	"mlcode/decision_tree"
	"mlcode/neural_net"
	"mlcode/regression"
	"os"
)

func main() {

	// Get selection from command line
	arg := "dectree" // default if none provided
	//arg = "forest"
	if len(os.Args) > 1 {
		arg = os.Args[1]
	}

	// Run selected demo, or show error message
	if arg == "linear" {
		fmt.Println("Running linear regression demo")
		regression.Linear_Regression_Demo()
	} else if arg == "logistic" {
		fmt.Println("Running logistic regression demo")
		regression.Logistic_Regression_Demo()
	} else if arg == "neural" {
		fmt.Println("Running MNIST demo")
		neural_net.MnistDemo()
	} else if arg == "dectree" {
		fmt.Println("Running decision tree demo (titanic)")
		decision_tree.DecisionTreeDemo2()
	} else if arg == "forest" {
		fmt.Println("Running random forest demo (titanic)")
		decision_tree.RandomForestDemo()
	} else {
		fmt.Println("Specify: linear, logistic, neural, dectree, or forest")
	}
}
