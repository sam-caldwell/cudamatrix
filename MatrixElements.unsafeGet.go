package cudamatrix

// unsafeGet - given a (row,col) coordinate, return the value or an error
//
//go:inline
func (m *MatrixElements) unsafeGet(row, col uint) (value float64) {

	return (*m)[row][col]

}
