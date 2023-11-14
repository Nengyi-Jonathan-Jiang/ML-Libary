package matrix.exception;

import matrix.Matrix;

public class MatrixRowAccessException extends MatrixException {
    public MatrixRowAccessException(Matrix matrix, int row) {
        super("Invalid matrix access: Tried to access row %d of <%d, %d>"
                .formatted(row, matrix.rows(), matrix.columns()));
    }
}
