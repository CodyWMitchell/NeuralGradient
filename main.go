// Some experiments using a neural network to generate a complex gradient
// By: Cody Mitchell

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"time"

	"github.com/goml/gobrain"
)

var width, height = 1920, 1080
var iterations, hiddenLayers, learningRate, mutationFactor = 100000, 10, 0.2, 0.1
var saveFilename = "gradient.png"

// define the training set
// inputs : the x y coordinates of the point
// outputs : the color of the point (red, green, blue)
var patterns = [][][]float64{
	{{1, 1}, {0, 0, 0}},
	{{0, 0}, {1, 0.2, 0}},
	{{0.5, 0.5}, {1, 0, 0}},
	{{0.5, 0.55}, {1, 1, 1}},
	{{0.5, 0.6}, {1, 0, 0}},
	{{0.4, 0.6}, {1, 0.2, 0}},
	{{0.4, 0.4}, {0, 0.2, 0}},
	{{0.2, 0.8}, {0, 0.2, 0.7}},
	{{0.8, 0.8}, {0, 0.2, 0}},
	{{0.8, 0.2}, {1, 1, 1}},
	{{0.2, 0.2}, {0, 0.2, 0}},
	{{0.2, 0.8}, {1, 0, 0}},
	{{0.2, 0.8}, {1, 1, 0.7}},
	{{0.8, 0.2}, {0, 0, 0}},
	{{0.2, 0.2}, {1, 0.2, 1}},
	{{0, 0}, {0, 0.2, 0}},
	{{0, 0}, {0, 0.2, 0}},
	{{0, 0}, {0, 0.2, 0}},
}

func randomizeArray(array [][][]float64) {
	// randomize the array of positions and colors
	for i := 0; i < len(array); i++ {
		for j := 0; j < len(array[i]); j++ {
			for k := 0; k < len(array[i][j]); k++ {
				array[i][j][k] = float64(rand.Intn(100)) / 100
			}
		}
	}
}

func main() {
	t := time.Now().UnixNano()
	rand.Seed(t)
	randomizeArray(patterns)

	var img = image.NewRGBA(image.Rect(0, 0, width, height))

	ff := &gobrain.FeedForward{}
	ff.Init(2, hiddenLayers, 3)
	ff.Train(patterns, iterations, learningRate, mutationFactor, true)

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			inputs := []float64{float64(x) / float64(width), float64(y) / float64(height)}
			outputs := ff.Update(inputs)
			col := color.RGBA{uint8(outputs[0] * 255), uint8(outputs[1] * 255), uint8(outputs[2] * 255), 255}

			img.Set(x, y, col)
		}
	}

	f, err := os.Create(saveFilename)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	png.Encode(f, img)
	fmt.Println("SEED", t)
}
