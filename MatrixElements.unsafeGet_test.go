package cudamatrix

import (
	"testing"
)

func TestMatrixElements_unsafeGet(t *testing.T) {
	assertPanic := func(t *testing.T, f func(), expected string) {
		t.Helper()
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("expected panic, but function completed normally")
			}
		}()

		f()
	}
	t.Run("Empty: expect panic if we try to use unsafeGet()", func(t *testing.T) {
		m := MatrixElements{}
		assertPanic(t, func() {
			if v := m.unsafeGet(0, 0); v != 0 {
				t.Fatalf("expected panic, not a value evaluation")
			}
		}, "runtime error: index out of range [0] with length 0")
	})
	t.Run("Not Empty: expect panic if we try to use unsafeGet() out of bounds", func(t *testing.T) {
		m := MatrixElements{
			{0, 1, 2, 3},
			{0, 1, 2, 3},
			{0, 1, 2, 3},
			{0, 1, 2, 3},
		}
		assertPanic(t, func() {
			if v := m.unsafeGet(10, 10); v != 0 {
				t.Fatalf("expected panic, not a value evaluation")
			}
		}, "runtime error: index out of range [0] with length 0")
	})

	t.Run("Not Empty: expect no error if we get elements within bounds of the matrix", func(t *testing.T) {
		m := MatrixElements{
			{0, 1, 2, 3},
			{0, 1, 2, 3},
			{0, 1, 2, 3},
			{0, 1, 2, 3},
		}
		for col := uint(0); col < m.cols(); col++ {
			for row := uint(0); row < m.rows(); row++ {
				if v := m.unsafeGet(row, col); uint(v) != col {
					t.Fatalf("(%d,%d) expected %d, got %d", row, col, col, uint(v))
				}
			}
		}
	})

}
