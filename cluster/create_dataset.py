#!/bin/python
#
# Create a dummy data set n clusters of 2D points, redirect to
# file to create CSV file

from sklearn.datasets import make_blobs

# Create a dataset of 2D distributions
nclusters = 5
XY, labels = make_blobs(n_samples = 1000, centers = nclusters)

# Write as CSV file
print('"X","Y"')
for r in XY.tolist():
    print('%.4f,%4f' % tuple(r))
