package cudamatrix

import (
	"fmt"
	"github.com/sam-caldwell/errors"
)

// get - given a (row,col) coordinate, return the value or an error
func (m *MatrixElements) get(row, col uint) (value float64, err error) {

	if row < m.rows() && col < m.cols() {
		return m.unsafeGet(row, col), nil
	}

	return 0, fmt.Errorf(errors.BoundsCheckError)

}
