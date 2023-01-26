// kmeans.go
//
// Clustering, using KMeans

package cluster

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"mlcode/utils"
)

// Maximum number of iterations
var maxIterations int = 1000

// Perform KMeans clustering on a dataset, return list of labels
func KMeans(df *utils.DataFrame, nclust int) []int {

	// We only want the numeric columns, as floats, so convert to matrix
	m := df.ToMatrix()
	nr, nc := m.Dims()
	if nr < 2 || nc < 2 {
		fmt.Println("Not enough data")
		return []int{}
	}

	// Initialize array of centroids
	centroids := make([][]float64, nclust, nclust)
	for i := 0; i < nclust; i++ {
		centroids[i] = make([]float64, nc, nc)
	}

	// Initialize array of cluster assignments, with each row initially
	// assigned to a random cluster
	clusters := make([]int, nr, nr)
	for i := 0; i < nr; i++ {
		clusters[i] = rand.Intn(nclust)
	}
	fmt.Println("Initial assignments:", clusters)

	// Begin iterations
	iter := 0
	counts := make([]int, nclust, nclust)
	for {

		// Calculate the centroid of each cluster, i.e., the average x/y position
		for i := 0; i < nclust; i++ { // zero-out the centroids
			counts[i] = 0
			for j := 0; j < nc; j++ {
				centroids[i][j] = 0
			}
		}
		for i := 0; i < nr; i++ {
			r := m.RowView(i)
			clust := clusters[i] // current cluster for this row
			counts[clust]++      // incr count of rows for this cluster
			for j := 0; j < nc; j++ {
				centroids[clust][j] += r.AtVec(j)
			}
		}
		for i := 0; i < nclust; i++ {
			for j := 0; j < nc; j++ {
				if counts[i] > 0 {
					centroids[i][j] /= float64(counts[i])
				}
			}
		}
		fmt.Println("Iteration", iter, ": Centroids =", centroids)

		// Assign each row to the closest cluster
		moved := false
		for ri := 0; ri < nr; ri++ {
			r := m.RowView(ri)
			c := closestCluster(r, centroids)
			if clusters[ri] != c {
				clusters[ri] = c
				moved = true
			}
		}

		// Stop when no more movement, or max iterations reached
		iter++
		if !moved || iter > maxIterations {
			break
		}
	}

	// Return final cluster assignments
	return clusters
}

// Find the index of the closest cluster centroid for a row
func closestCluster(r mat.Vector, centroids [][]float64) int {
	var closest int
	var dMin float64
	for ci := 0; ci < len(centroids); ci++ {
		d := distance(r, centroids[ci])
		if ci == 0 || d < dMin {
			dMin = d
			closest = ci
		}
	}
	return closest
}

// Calculate the Euclidean distance between two points, in any
// number of dimensions
func distance(r mat.Vector, p []float64) float64 {
	var d float64
	for i := 0; i < len(p); i++ {
		d += math.Pow(r.AtVec(i)-p[i], 2)
	}
	return math.Sqrt(d)
}
