package cudamatrix

func (lhs *Matrix) Cols() uint {
	return lhs.data.cols()
}
