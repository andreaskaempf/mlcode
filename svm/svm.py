import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


# Feature selection: remove correlated features
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped

# Remove less significant features
def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

# Compute cost, for model training
def compute_cost(W, X, Y):

    # Calculate distances
    distances = 1 - Y * np.dot(X, W)
    distances[distances < 0] = 0  # i.e., max(0, distance)
    print('***', W.shape, X.shape, Y.shape, distances.shape)
    
    # calculate hinge loss
    N = X.shape[0]
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


# I haven't tested it but this same function should work for
# vanilla and mini-batch gradient descent as well
def calculate_cost_gradient(W, X_batch, Y_batch):

    print('W:', W, W.shape)
    print('X:', X_batch, X_batch.shape)
    print('Y:', Y_batch, Y_batch.shape)

    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        #print('Converting to arrays:', X_batch.shape, Y_batch.shape)
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array
        #print('  =>', X_batch.shape, Y_batch.shape)

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    return dw


# Stochastic Gradient Descent
def sgd(features, outputs):

    max_epochs = 5000
    weights = np.zeros(features.shape[1]) # row of 13 zeros
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent

    print("Initial cost:", compute_cost(weights, features, outputs))
        
    # stochastic gradient descent
    for epoch in range(1, max_epochs):

        # TODO: Shuffle to prevent repeating update cycles (but lowers 
        # performance a bit)
        # X:(455, 13)
        #X, Y = shuffle(features, outputs)
        X, Y = features, outputs

        # Process each row of features, ascent is a vector of 13 numbers
        for ind, x in enumerate(X): # each row number, row data
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        # Stop when no longer improving
        cost = compute_cost(weights, features, outputs)
        print("Epoch %d: cost = %.2f" % (epoch, cost))
        if abs(prev_cost - cost) < cost_threshold * prev_cost:
            return weights
        prev_cost = cost

    return weights

# set hyper-parameters
regularization_strength = 10000
learning_rate = 0.000001

# read data in pandas (pd) data frame
print("reading dataset...")
data = pd.read_csv('../data/breastcancer.csv')

# drop last column (extra column added by pd)
# and unnecessary first column (id)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

# convert categorical labels to numbers
print("Feature engineering...")
diag_map = {'M': 1.0, 'B': -1.0}
data['diagnosis'] = data['diagnosis'].map(diag_map)

# put features & outputs in different data frames
Y = data.loc[:, 'diagnosis']
X = data.iloc[:, 1:]

# TODO: filter features
#remove_correlated_features(X)
#remove_less_significant_features(X, Y)

# normalize data for better convergence and to prevent overflow
X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

# insert 1 in every row for intercept b
# X: 569 rows x 13 cols (last is intercept 1)
# Y: vector of 569 values (1/-1)
X.insert(loc=len(X.columns), column='intercept', value=1)

# split data into train and test set
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

# train the model
print("Training ...")
W = sgd(X_train.to_numpy(), y_train.to_numpy())
print("Weights:", W)

# test the model
print("Testing ...")
y_train_predicted = np.array([])
for i in range(X_train.shape[0]):
    yp = np.sign(np.dot(X_train.to_numpy()[i], W))
    y_train_predicted = np.append(y_train_predicted, yp)

y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = np.sign(np.dot(X_test.to_numpy()[i], W))
    y_test_predicted = np.append(y_test_predicted, yp)

print("Accuracy on test dataset:", accuracy_score(y_test, y_test_predicted))
print("Recall on test dataset:", recall_score(y_test, y_test_predicted))
print("Precision on test dataset:", recall_score(y_test, y_test_predicted))

