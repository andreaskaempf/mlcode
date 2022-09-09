import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

def classify(X, w):
    return np.round(forward(X, w))

def loss(X, Y, w):

    #print('*** loss:\nX =', X, X.shape)
    #print('Y =', Y, Y.shape)
    #print('w =', w, w.shape)

    y_hat = forward(X, w)
    #print('y_hat =', y_hat, y_hat.shape)

    t1 = Y * np.log(y_hat)
    #print('t1 =', t1, t1.shape)

    t2 = (1 - Y) * np.log(1 - y_hat)
    #print('t2 =', t2, t2.shape)

    return -np.average(t1 + t2)

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))

# Prepare data
#x1, x2, x3, y = np.loadtxt("data/police.txt", skiprows=1, unpack=True)
#X = np.column_stack((np.ones(x1.size), x1, x2, x3))
#Y = y.reshape(-1, 1)
data = pd.read_csv('data/police.txt').values

# Split into X and Y
X = data[:,:3]
Y = data[:,3]
Y = Y.reshape((-1,1)) # One column
print('X =', X)
print('Y =', Y)

# Train model
w = train(X, Y, iterations=10000, lr=0.001)
print(w)

# Test it
test(X, Y, w)
