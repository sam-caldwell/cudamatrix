package cudamatrix

// cols - Return the number of columns in the MatrixElements
func (m *MatrixElements) cols() uint {
	if m.rows() == 0 {
		return 0
	}
	return uint(len((*m)[0]))
}
