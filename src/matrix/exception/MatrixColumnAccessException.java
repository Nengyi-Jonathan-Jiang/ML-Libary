package matrix.exception;

import matrix.Matrix;

public class MatrixColumnAccessException extends MatrixException {
    public MatrixColumnAccessException(Matrix matrix, int column) {
        super("Invalid matrix access: Tried to access column %d of <%d, %d>"
                .formatted(column, matrix.rows(), matrix.columns()));
    }
}
