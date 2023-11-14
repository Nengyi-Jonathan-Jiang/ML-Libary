package matrix.exception;

import matrix.Matrix;

public class MatrixDimensionMismatchException extends MatrixDimensionException {
    public MatrixDimensionMismatchException(Matrix A, Matrix B) {
        super("Matrix Dimension mismatch: <" + A.rows() + ", " + A.columns() + "> != <" + B.rows() + ", " + B.columns() + ">");
    }
    public MatrixDimensionMismatchException(Matrix A, int rows, int columns) {
        super("Matrix Dimension mismatch: <" + A.rows() + ", " + A.columns() + "> != <" + rows + ", " + columns + ">");
    }
    public MatrixDimensionMismatchException(String why) {
        super(why);
    }
}
