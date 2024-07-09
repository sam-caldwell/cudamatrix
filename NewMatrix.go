package cudamatrix

import (
	"fmt"
	"github.com/sam-caldwell/errors"
)

// NewMatrix creates a new matrix with the given dimensions.
func NewMatrix(rows, cols uint) (result *Matrix, err error) {

	var m Matrix

	if rows == 0 || cols == 0 {
		return nil, fmt.Errorf(errors.BoundsCheckError)
	}

	m.data = make(MatrixElements, rows)

	for row := uint(0); row < m.data.rows(); row++ {

		m.data[row] = make([]float64, cols)

	}

	return &m, nil

}
