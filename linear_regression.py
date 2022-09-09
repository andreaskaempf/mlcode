# Regression.py
# Multi-variate linear regression, adapted from chapters 1-4 of "ML fur
# Softwareentwickler"

import numpy as np
import pandas as pd  # only for reading data

# Read data
# Columns: Reservations, Temp, Tourists, Pizzas
#data = np.loadtxt('pizza_3_vars.txt', skiprows = 1)
data = pd.read_csv('pizza_3_vars.txt').values

# Split into X and Y
X = data[:,:3]
Y = data[:,3]
Y = Y.reshape((-1,1)) # One column
print('X =', X)
print('Y =', Y)


# Predict Y values (one column), given X values (one column per variable) and
# coefficients (vector of values, one per X column)
def predict(X, w):
    return np.matmul(X, w)

# Calcualte the mean squared difference between predicted and actual values
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

# Compute the gradient
def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

# Train the model using gradient descent on coeffients, until loss
# stops improving by at least the tolerance
def train(X, Y, lr, tol):
    w = np.zeros((X.shape[1], 1))
    prevLoss = 0
    for i in range(1000):  # maximum iterations
        l = loss(X, Y, w)
        print('Iteration %d: loss = %f' % (i, l))
        if i > 0 and abs(prevLoss - l) < tol:
            print('Solution found')
            break
        prevLoss = l
        w -= gradient(X, Y, w) * lr
    return w


# Train the regression model
w = train(X, Y, .001, .001)
print(w)
