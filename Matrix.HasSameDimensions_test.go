package cudamatrix

import "testing"

func TestMatrix_HasSameDimensions(t *testing.T) {
	t.Run("Empty Matrices", func(t *testing.T) {
		A := Matrix{}
		B := Matrix{}
		if !A.HasSameDimensions(&B) {
			t.Fatal("expected the same dimensions.  Got different.")
		}
	})
}
