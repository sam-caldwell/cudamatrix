package cudamatrix

import (
	"fmt"
	"github.com/sam-caldwell/errors"
	"testing"
)

func TestMatrix_Multiply(t *testing.T) {
	compareMatrix := func(expected, C *Matrix) (r uint, c uint, ok bool) {
		for r = uint(0); r < C.Rows(); r++ {
			for c = uint(0); c < C.Cols(); c++ {
				if C.data[r][c] != expected.data[r][c] {
					return r, c, false
				}
			}
		}
		return 0, 0, true
	}

	dumpMatrix := func(M *Matrix) {
		var v float64
		fmt.Print("----\n")
		for r := uint(0); r < M.Rows(); r++ {
			fmt.Printf("| ")
			for c := uint(0); c < M.Cols(); c++ {
				v, _ = M.Get(r, c)
				fmt.Printf("%04.1f ", v)
			}
			fmt.Print(" |\n")
		}
		fmt.Print("----\n")
	}

	t.Run("Multiply lhs against nil rhs matrix", func(t *testing.T) {
		var A Matrix  // empty
		var B *Matrix // nil
		if C, err := A.Multiply(B); err == nil {
			t.Fatalf("error expected.  none found")
		} else {
			if err.Error() != errors.NilPointer {
				t.Fatalf("expected NilPointer error. Got: %v", err)
			}
			if C != nil {
				t.Fatalf("expect nil result on error")
			}
		}
	})
	t.Run("Multiply two empty matrices", func(t *testing.T) {
		var A Matrix // empty
		var B Matrix // empty
		if C, err := A.Multiply(&B); err == nil {
			t.Fatal("expected an error. got none")
		} else {
			if err.Error() != errors.BoundsCheckError {
				t.Fatalf("unexpected error.  Got %v", err)
			}
			if C != nil {
				t.Fatal("expect nil result on error")
			}
		}
	})
	t.Run("Multiply two matrices where lhs cols != rhs rows. expect error", func(t *testing.T) {
		/*
		 * Assumptions:
		 *     A(4x4)
		 *     B(3x4)
		 */
		var (
			A *Matrix
			B *Matrix
		)
		t.Run("Setup the test", func(t *testing.T) {
			t.Run("Matrix A", func(t *testing.T) {
				A = &(Matrix{
					data: MatrixElements{
						{0, 1, 2, 3},
						{4, 5, 6, 7},
						{8, 9, 10, 11},
						{12, 13, 14, 15},
					},
				})
			})
			t.Run("Matrix B", func(t *testing.T) {
				B = &(Matrix{
					data: MatrixElements{
						{0, 1, 2, 3},
						{4, 5, 6, 7},
						{8, 9, 10, 11},
					},
				})
			})
		})
		t.Run("Perform Multiplication.  Expect error", func(t *testing.T) {
			if C, err := A.Multiply(B); err == nil {
				t.Fatal("expected an error. got none")
			} else {
				if err.Error() != errors.MatrixDimensionMismatch {
					t.Fatalf("unexpected error.  Got %v", err)
				}
				if C != nil {
					t.Fatal("expect nil result on error")
				}
			}
		})
	})
	t.Run("Multiply two matrices of same dimensions (4x4)", func(t *testing.T) {
		var (
			A   *Matrix
			B   *Matrix
			C   *Matrix
			err error
		)
		t.Run("Setup the test", func(t *testing.T) {
			t.Run("Matrix A", func(t *testing.T) {
				A = &(Matrix{
					data: MatrixElements{
						{0, 1, 2, 3},
						{4, 5, 6, 7},
						{8, 9, 10, 11},
						{12, 13, 14, 15},
					},
				})
			})
			t.Run("Matrix B", func(t *testing.T) {
				B = &(Matrix{
					data: MatrixElements{
						{0, 1, 2, 3},
						{4, 5, 6, 7},
						{8, 9, 10, 11},
						{12, 13, 14, 15},
					},
				})
			})
		})
		t.Run("Perform Multiplication.", func(t *testing.T) {
			if C, err = A.Multiply(B); err != nil {
				t.Fatal(err)
			}
		})
		t.Run("Evaluate results", func(t *testing.T) {
			expected := Matrix{
				data: MatrixElements{
					{56.0, 62.0, 68.0, 74.0},
					{152.0, 174.0, 196.0, 218.0},
					{248.0, 286.0, 324.0, 362.0},
					{344.0, 398.0, 452.0, 506.0},
				},
			}
			dumpMatrix(C)
			if r, c, ok := compareMatrix(&expected, C); !ok {
				t.Fatalf("compare failed at (%d,%d)", r, c)
			}
		})
	})
	t.Run("Multiply two matrices of different dimensions where lhs cols == rhs rows. ok", func(t *testing.T) {
		//var (
		//	A Matrix
		//	B Matrix
		//)
		t.Run("Setup the test", func(t *testing.T) {

		})
		t.Run("Perform Multiplication.", func(t *testing.T) {

		})
		t.Run("Evaluate results", func(t *testing.T) {

		})
	})
}
