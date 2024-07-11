package cudamatrix

import (
	"testing"
)

func TestMatrix_from1dArray(t *testing.T) {
	t.Run("Empty Matrix", func(t *testing.T) {
		var linearMatrix []float64
		matrix := &Matrix{
			data: make([][]float64, len(linearMatrix)),
		}
		if matrix.Rows() != 0 || matrix.Cols() != 0 {
			t.Fatalf("expected empty output\n"+
				"rows: %d\n"+
				"cols: %d", matrix.Rows(), matrix.Cols())
		}
	})
	t.Run("Happy path: 3x3 matrix", func(t *testing.T) {
		// Create a sample 1D array
		flatData := []float64{
			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0,
		}

		// Expected 2D matrix
		expectedData := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
		}

		// Create a matrix to hold the result
		matrix := &Matrix{
			data: make([][]float64, 3),
		}

		// Initialize the matrix with empty rows
		for i := range matrix.data {
			matrix.data[i] = make([]float64, 3)
		}

		// Call the from1dArray method
		err := matrix.from1dArray(&flatData)
		if err != nil {
			t.Fatalf("from1dArray returned an error: %v", err)
		}

		// Compare the result with the expected matrix
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				if matrix.data[r][c] != expectedData[r][c] {
					t.Fatalf("from1dArray result does not match expected data.\n"+
						"(r,c): (%d,%d)\n"+
						"  Got: %v\n"+
						" Want: %v", r, c, matrix.data, expectedData)
				}
			}
		}
	})

	t.Run("Happy path: 4x3 matrix", func(t *testing.T) {
		// Create a sample 1D array
		flatData := []float64{
			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0,
			10.0, 11.0, 12.0,
		}

		// Expected 2D matrix
		expectedData := [][]float64{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
			{10.0, 11.0, 12.0},
		}

		// Create a matrix to hold the result
		matrix := &Matrix{
			data: make([][]float64, 4),
		}

		// Initialize the matrix with empty rows
		for i := range matrix.data {
			matrix.data[i] = make([]float64, 3)
		}

		// Call the from1dArray method
		err := matrix.from1dArray(&flatData)
		if err != nil {
			t.Fatalf("from1dArray returned an error: %v", err)
		}

		// Compare the result with the expected matrix
		for r := 0; r < 4; r++ {
			for c := 0; c < 3; c++ {
				if matrix.data[r][c] != expectedData[r][c] {
					t.Fatalf("from1dArray result does not match expected data.\n"+
						"(r,c): (%d,%d)\n"+
						"  Got: %v\n"+
						" Want: %v", r, c, matrix.data, expectedData)
				}
			}
		}
	})
}
