package cudamatrix

import "testing"

func TestMatrix_Cols(t *testing.T) {
	t.Run("Empty: zero-rows has zero columns", func(t *testing.T) {
		m := Matrix{}
		if m.Cols() != 0 {
			t.Fatal("expect zero cols on empty")
		}
	})
	t.Run("Has Rows but no columns", func(t *testing.T) {
		m := Matrix{}
		m.data = make([][]float64, 13)
		if m.Cols() != 0 {
			t.Fatal("expect zero cols because none were declared")
		}
	})
	t.Run("Has Rows and columns", func(t *testing.T) {
		m := Matrix{}
		m.data = make([][]float64, 13)
		for row := uint(0); row < m.data.rows(); row++ {
			m.data[row] = make([]float64, 20)
		}
		if c := m.Cols(); c != 20 {
			t.Fatalf("expected 20 cols. got %d cols", c)
		}
	})
}
