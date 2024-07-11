package cudamatrix

import "testing"

func TestMatrix_Rows(t *testing.T) {
	t.Run("Empty: zero-rows has zero columns", func(t *testing.T) {
		m := Matrix{}
		if m.Rows() != 0 {
			t.Fatal("expect zero rows on empty")
		}
	})
	t.Run("Has Rows but no columns", func(t *testing.T) {
		m := Matrix{}
		m.data = make([][]float64, 13)
		if m.Rows() != 13 {
			t.Fatalf("expect 13 rows. got %d", m.Rows())
		}
	})
	t.Run("Has Rows and columns", func(t *testing.T) {
		m := Matrix{}
		m.data = make([][]float64, 13)
		for row := uint(0); row < m.Rows(); row++ {
			m.data[row] = make([]float64, 20)
		}
		if r := m.Rows(); r != 13 {
			t.Fatalf("expected 13 rows. got %d rows", r)
		}
	})
}
