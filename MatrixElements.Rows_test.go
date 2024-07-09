package cudamatrix

import "testing"

func TestMatrixElements_Rows(t *testing.T) {
	t.Run("Empty: zero-rows has zero columns", func(t *testing.T) {
		m := MatrixElements{}
		if m.rows() != 0 {
			t.Fatal("expect zero rows on empty")
		}
	})
	t.Run("Has Rows but no columns", func(t *testing.T) {
		m := MatrixElements{}
		m = make([][]float64, 13)
		if m.rows() != 13 {
			t.Fatalf("expect 13 rows. got %d", m.rows())
		}
	})
	t.Run("Has Rows and columns", func(t *testing.T) {
		m := MatrixElements{}
		m = make([][]float64, 13)
		for row := uint(0); row < m.rows(); row++ {
			m[row] = make([]float64, 20)
		}
		if r := m.rows(); r != 13 {
			t.Fatalf("expected 13 rows. got %d rows", r)
		}
	})
}
