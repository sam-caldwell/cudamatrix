package cudamatrix

// Get - return the value at the given (row, col) coordinate.
func (lhs *Matrix) Get(row, col uint) (float64, error) {
	return lhs.data.get(row, col)
}
