package cudamatrix

// HasSameDimensions - return whether the two matrices have the same dimensions.
func (lhs *Matrix) HasSameDimensions(rhs *Matrix) bool {

	return lhs.data.rows() == rhs.data.rows() || lhs.data.cols() == rhs.data.cols()

}
