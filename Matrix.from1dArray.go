package cudamatrix

import (
	"fmt"
	"github.com/sam-caldwell/errors"
)

// from1dArray converts a flat 1D array to its 2D equivalent.
//
// This assumes that the 1D array contains the same
// number of elements as the 2D array.
func (lhs *Matrix) from1dArray(flatData *[]float64) error {

	// number of columns and rows in the 2D array
	rows := lhs.data.rows()
	cols := lhs.data.cols()

	// Ensure the length of the input matches the expected size
	if uint(len(*flatData)) != cols*rows {
		return fmt.Errorf(errors.MatrixDimensionMismatch)
	}
	for row := uint(0); row < rows; row++ {
		copy((*lhs).data[row], (*flatData)[row*cols:(row+1)*cols])
	}

	return nil
}
