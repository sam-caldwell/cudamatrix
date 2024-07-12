package cudamatrix

// flatten - flatten the 2D matrix into a 1D array
func (lhs *Matrix) flatten() *[]float64 {

	rows := lhs.data.rows()
	cols := lhs.data.cols()

	flatResult := make([]float64, rows*cols)

	for r := uint(0); r < rows; r++ {

		copy(flatResult[r*cols:], (*lhs).data[r])

	}

	return &flatResult

}
