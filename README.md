# Machine learning and data science in Go

This is a simple machine learning library in Go, which uses gonum for matrices.
It is a set of focused utility classes, and not intended to replace golearn,
goml, or other mature Go libraries.

For now, it only handles data in numeric (float64) matrices, and there are utility
functions in matrices.go to read CSV files and extract columns (see examples below).

For the decision tree, I added a simple DataFrame implementation, and will probably
move the other algorithms over to this in time.

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

Only does classification for now, using floating point columns in a dataframe,
label has to be a string column. See example using Iris data set:

    // Build and train a tree
	df, _rr := dataframe.ReadCSV("data/iris.csv")
	tree := DecisionTree(df, "variety", 0)  // uses "variety" as the label
	PrintTree(tree, 0)  // start indentation at level zero

	// Make predictions
    row := df.GetRow(5)         // get dataframe with just row 5
    pred := Predict(tree, row) // returns predicted label

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
