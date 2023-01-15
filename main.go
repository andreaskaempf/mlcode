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
	arg := "undefined"
	if len(os.Args) > 1 {
		arg = os.Args[1]
	}
	//arg = "dectree" // uncomment for debugging

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
		fmt.Println("Running decision tree demo")
		decision_tree.DecisionTreeDemo()
	} else {
		fmt.Println("Specify: linear, logistic, neural, or dectree")
	}
}
