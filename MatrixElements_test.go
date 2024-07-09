package cudamatrix

import (
	"reflect"
	"testing"
)

func TestMatrixElementsType(t *testing.T) {
	// Create a sample MatrixElements instance
	var elements MatrixElements = [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	// Check if the type of elements is as expected
	expectedType := "cudamatrix.MatrixElements"
	actualType := reflect.TypeOf(elements).String()

	if actualType != expectedType {
		t.Errorf("expected type %s, but got %s", expectedType, actualType)
	}

	// Additional check: ensure the elements contain the correct type
	if len(elements) > 0 {
		if reflect.TypeOf(elements[0][0]).Kind() != reflect.Float64 {
			t.Errorf("expected element type float64, but got %s", reflect.TypeOf(elements[0][0]).Kind())
		}
	}
}
