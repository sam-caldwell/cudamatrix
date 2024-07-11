package cudamatrix

func (lhs *Matrix) Rows() uint {
	return lhs.data.rows()
}
