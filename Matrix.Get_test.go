package cudamatrix

import (
	"github.com/sam-caldwell/errors"
	"testing"
)

func TestMatrix_Get(t *testing.T) {
	t.Run("Empty: expect panic if we try to use unsafeGet()", func(t *testing.T) {
		m := Matrix{}
		if v, err := m.Get(0, 0); err == nil {
			t.Fatalf("expected BoundsCheckError, got none")
		} else {
			if err.Error() != errors.BoundsCheckError {
				t.Fatalf("expected BoundsCheckError.  Got %v", err)
			}
			if v != 0 {
				t.Fatal("expect 0 on error")
			}
		}
	})
	t.Run("Not Empty: expect panic if we try to use unsafeGet() out of bounds", func(t *testing.T) {
		m := Matrix{
			data: MatrixElements{
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
			},
		}
		if v, err := m.Get(10, 10); err == nil {
			t.Fatalf("expected BoundsCheckError, got none")
		} else {
			if err.Error() != errors.BoundsCheckError {
				t.Fatalf("expected BoundsCheckError.  Got %v", err)
			}
			if v != 0 {
				t.Fatal("expect 0 on error")
			}
		}
	})

	t.Run("Not Empty: expect no error if we get elements within bounds of the matrix", func(t *testing.T) {
		m := Matrix{
			data: MatrixElements{
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
			},
		}
		for col := uint(0); col < m.data.cols(); col++ {
			for row := uint(0); row < m.data.rows(); row++ {
				if v, err := m.Get(2, 2); err != nil {
					t.Fatalf("expected no error.  Got %v", err)
				} else {
					if v != 2 {
						t.Fatalf("value mismatch.  expected 2. got %f", v)
					}
				}
			}
		}
	})

}
