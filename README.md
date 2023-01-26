# Machine learning and data science in Go

This is a simple machine learning library in Go, which uses gonum for matrices.
It is a set of focused utility classes with demonstration functions, and not
intended to replace golearn, goml, or other mature Go libraries.

A simple DataFrame implementation is included, to read CSV files and allow
for categorical variables (decision trees and random forest only).


## Linear Regression

Sample usage (also see unit test file, some functions have been moved into modules, 
and these examples need to be updated):


    data, _ := ReadMatrixCSV("data/police.txt")
    X := extractCols(data, 0, 2) // all cols except last
    Y := extractCols(data, 3, 3) // just the last col

    m := LinearRegression{}     // create model
    m.Verbose = true            // set flag (also lr, iterations, tol)
    m.Train(X, Y)               // train the  model
    MatPrint(m.w)               // prints final coefficients
    preds := m.Predict(X)       // make prediction

## Logistic Regression

Sample usage (also see unit test file):

    m := LogisticRegression{}   // create model
    m.Verbose = true            // set flag (also lr, iterations, tol)
    m.Train(X, Y)               // train the  model
    MatPrint(m.w)               // prints final coefficients
    preds := m.Predict(X)       // make prediction

## Decision Tree

Only does classification for now, using integer, floating point, or categorical
(string) columns in a dataframe. The label you are training on has to be a string 
column. See example using Iris data set (there is a second demo using the Titanic
data set):

    // Build and train a tree
	df, _rr := dataframe.ReadCSV("data/iris.csv")
	tree := DecisionTree(df, "variety", 0)  // uses "variety" as the label
	PrintTree(tree, 0)  // start indentation at level zero

	// Make predictions
    row := df.GetRow(5)         // get dataframe with just row 5
    pred := Predict(tree, row) // returns predicted label

## Random Forest

Uses bagging (random sampling of data with replacement) to train a group
of decision trees (no randomization of features yet), then predicts by
taking the most common prediction from among the trees. Demo uses the
Titanic data set. Sample usage:

    // Create a random forest of 200 trees
	forest := RandomForest(df, "Survived", 200)

	// Make a prediction from one row
    row := df.GetRow(5)
    pred := RandomForestPredict(forest, row)

## Support Vector Machine

Simple implementation using using stochastic gradient descent, based on
[this article](https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2).
See demo function in svm.go, usage as follows:

	// Read the breast cancer dataset, see demo to
	// remove ID and diagnosis columns, normalize, 
    // convert diagnosis to 1/-1, add intercept, and
    // convert to matrices
	df, _ := utils.ReadCSV("data/breastcancer.csv")

	// Train the model, returns final weights
	W := sgd(X, Y)

	// Make predictions (need to take just sign of results)
	nr, _ := X.Dims()
	preds := mat.NewVecDense(nr, nil)
	preds.MulVec(X, W)

## K-Means Clustering

Implementation based on my recollection of the algorithm. Clusters are
initialized randomly, then centroids calculated and rows assigned to the
cluster with the nearest centroid, repeatedly until there is no movement. Demo
creates a colour-coded scatterplot, using GoNum's plot library. Should work
with any number of dimensions, demo uses only two.  You pass it a dataframe,
and it uses all available numeric columns.

Sample usage:


	// Read a data set and divide it into 5 clusters
	df, _ := utils.ReadCSV("clusters2D.csv")
	clusters := KMeans(df, 5) // returns slice of cluster numbers


## Neural Network

Simple 3-layer neural network, with one hidden layer, based on chapters 9-12 of
"Programming Machine Learning" by Paolo Perotta (book uses Python). Demo trains
against MNIST digits database, achieves 89% accuracy.


	// Train multi-class classifier using 10-column one-hot encoded labels
	m := regression.MultiLogRegression{Iterations: 100, LR: 1e-5, Verbose: true}
	m.Train(pics, labs1)

	// Predict on test data, see demo for simple accuracy measurement
	preds := m.Classify(tpics)


Andreas Kaempf, 2022-23
