package cudamatrix

// rows - return the number of rows in the MatrixElements
func (m *MatrixElements) rows() uint {
	return uint(len(*m))
}
