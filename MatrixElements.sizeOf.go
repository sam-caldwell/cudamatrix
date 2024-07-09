package cudamatrix

// sizeOf - Return the dimensions of the Matrix in (rows,columns)
func (m *MatrixElements) sizeOf() (rows, cols uint) {
	return m.rows(), m.cols()
}
