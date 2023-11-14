package matrix.exception;

import matrix.Matrix;

public class MatrixElementAccessException extends MatrixException {
    public MatrixElementAccessException(Matrix matrix, int row, int column) {
        super("Invalid matrix access: Tried to access %d, %d of <%d, %d>"
                .formatted(row, column, matrix.rows(), matrix.columns()));
    }
}
