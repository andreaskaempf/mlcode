package main

import (
	"fmt"
	"mlcode/neural_net"
	"mlcode/regression"
)

func main() {

	fmt.Println("Running linear regression demo")
	regression.Linear_Regression_Demo()

	fmt.Println("Running logistic regression demo")
	regression.Logistic_Regression_Demo()

	fmt.Println("Running MNIST demo")
	neural_net.MnistDemo()
}
