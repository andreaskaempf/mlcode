import numpy as np
import mnist as data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

def classify(X, w):
    y_hat = forward(X, w)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    print('\nloss:')
    print('X =', X)
    print('Y =', Y)
    print('w =', w)
    print('y_hat =', y_hat)
    print('first_term =', first_term)
    print('second_term =', second_term)
    print('first + second_term =', first_term + second_term)
    print('sum =', np.sum(first_term + second_term))
    return -np.sum(first_term + second_term) / X.shape[0]

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def report(i, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)
    matches = matches * 100.0 / Y_test.shape[0]
    training_loss = loss(X_train, Y_train, w)
    print("Iteration %d: loss %.6f, %.2f%%" % (i, training_loss, matches))

def train(X_train, Y_train, X_test, Y_test, iterations, lr):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(iterations):
        report(i, X_train, Y_train, X_test, Y_test, w)
        w -= gradient(X_train, Y_train, w) * lr
        print(w)
    report(iterations, X_train, Y_train, X_test, Y_test, w)
    return w

#train(data.X_train, data.Y_train, data.X_test, data.Y_test, 
#        iterations=3, lr=1e-5)

# Some tests: 5 instances, 2 classes
# X: 5 x 3
# w: 3 x 
# Y: 5 x 3
X = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]])
Y = np.array([[1,0], [1,0], [0,1], [1,0], [0,1]])
w = np.array([[.01, .02], [.03, .04], [.05, .06]])

print('X =', X)
print('Y =', Y)
print('w =', w)

f = forward(X, w)
print('forward(X,w) =')
print(f)

g = gradient(X, Y, w)
print('gradient(X, Y, w) =')
print(g)

w -= g * .01
print('adjusted w =')
print(w)

l = loss(X, Y, w)
print('loss =')
print(l)

