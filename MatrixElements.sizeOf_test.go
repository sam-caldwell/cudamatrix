package cudamatrix

import "testing"

func TestMatrixElements_SizeOf(t *testing.T) {
	t.Run("Empty: zero-rows has zero columns", func(t *testing.T) {
		m := MatrixElements{}
		if r, c := m.sizeOf(); r != 0 || c != 0 {
			t.Fatal("expect zero rows and columns on empty")
		}
	})
	t.Run("Has Rows but no columns", func(t *testing.T) {
		m := MatrixElements{}
		m = make([][]float64, 13)
		if r, c := m.sizeOf(); r != 13 || c != 0 {
			t.Fatal("expect zero rows and columns on empty")
		}
	})
	t.Run("Has Rows and columns", func(t *testing.T) {
		m := MatrixElements{}
		m = make([][]float64, 13)
		for row := uint(0); row < m.rows(); row++ {
			m[row] = make([]float64, 20)
		}
		if r, c := m.sizeOf(); r != 13 || c != 20 {
			t.Fatal("expect zero rows and columns on empty")
		}
	})
}
