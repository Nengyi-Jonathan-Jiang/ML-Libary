package matrix.exception;

import matrix.Matrix;

public class MatrixMultiplicationDimensionException extends MatrixDimensionMismatchException {
    public MatrixMultiplicationDimensionException(Matrix a, Matrix b) {
        super("Invalid matrix dimensions for multiplication: <%d, %d> x <%d, %d>".formatted(
            a.rows(), a.columns(), b.rows(), b.columns()
        ));
    }
}
