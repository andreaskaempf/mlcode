# Machine learning and data science in Go

This is a simple machine learning library in Go, which uses gonum for matrices.
It is a set of focused utility classes, and not intended to replace golearn,
goml, or other mature Go libraries.

For now, it only handles data in numeric (float64) matrices, and there are utility
functions in matrices.go to read CSV files and extract columns (see examples below).

## Linear Regression

Sample usage (also see unit test file):


    data, _ := readMatrixCSV("data/police.txt")
    X := extractCols(data, 0, 2) // all cols except last
    Y := extractCols(data, 3, 3) // just the last col

    m := LinearRegression{}     // create model
    m.verbose = true            // set flag (also lr, iterations, tol)
    m.train(X, Y)               // train the  model
    matPrint(m.w)               // prints final coefficients
    preds := m.predict(X)       // make prediction

## Logistic Regression

Sample usage (also see unit test file):

    m := LogisticRegression{}   // create model
    m.verbose = true            // set flag (also lr, iterations, tol)
    m.train(X, Y)               // train the  model
    matPrint(m.w)               // prints final coefficients
    preds := m.predict(X)       // make prediction



