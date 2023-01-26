// Demo function for K-Means clustering

package cluster

import (
	"fmt"
	"mlcode/utils"

	"gonum.org/v1/plot"

	"gonum.org/v1/plot/palette/brewer"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func KMeansDemo() {

	// Read dataset of five 2D clusters, created using sklearn.make_blobs
	df, err := utils.ReadCSV("data/clusters2D.csv")
	if err != nil {
		panic("Could not find data set")
	}

	// Divide into 5 clusters
	nclusters := 5
	clusters := KMeans(df, nclusters)
	fmt.Println("Final clusters:", clusters)

	// Create scatter plot
	graphPoints(df, clusters, nclusters)
}

// Create a scatter plot of points, color coded by cluster
func graphPoints(df *utils.DataFrame, clusters []int, nclusters int) {

	// Convert dataframe to matrix
	m := df.ToMatrix()
	nr, _ := m.Dims()

	// Create data points, first two columns only
	pts := make(plotter.XYs, nr)
	for i := 0; i < nr; i++ {
		r := m.RowView(i)
		pts[i].X = r.AtVec(0)
		pts[i].Y = r.AtVec(1)
	}

	// Create a plot
	p := plot.New()
	p.Add(plotter.NewGrid())

	// Create scatter plot from data points
	s, err := plotter.NewScatter(pts)
	if err != nil {
		panic("Unable to create scatter")
	}

	// Set up colors (range through brewer.QualitativePalettes to get palette
	// names: Set1, Set2, Set3, Accent, Dark2, Paired, Pastel1, Pastel2
	palette, err := brewer.GetPalette(0, "Dark2", nclusters)
	if err != nil {
		panic(err.Error())
	}
	colors := palette.Colors()

	// Function to set color for each point
	s.GlyphStyleFunc = func(i int) draw.GlyphStyle {
		clust := clusters[i]
		col := colors[clust]
		return draw.GlyphStyle{Color: col, Radius: vg.Points(3), Shape: draw.CircleGlyph{}}
	}

	// Add scatter to graph, and save as PNG
	p.Add(s)
	fmt.Println("Saving scatter plot to scatter.png")
	if p.Save(12*vg.Inch, 8*vg.Inch, "scatter.png") != nil {
		panic("Unable to save scatter")
	}
}
