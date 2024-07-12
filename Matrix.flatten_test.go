package cudamatrix

import (
	"reflect"
	"testing"
)

func TestMatrix_Flatten(t *testing.T) {
	// Create a sample 2D matrix
	matrixData := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	// Expected 1D array
	expectedFlatData := []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	}

	// Create a matrix with the sample data
	matrix := &Matrix{
		data: matrixData,
	}

	// Call the flatten method
	flatData := matrix.flatten()

	// Compare the result with the expected 1D array
	if !reflect.DeepEqual(*flatData, expectedFlatData) {
		t.Errorf("flatten result does not match expected data.\nGot: %v\nWant: %v", *flatData, expectedFlatData)
	}
}
