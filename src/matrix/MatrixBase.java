package matrix;

import matrix.exception.MatrixDimensionMismatchException;
import matrix.exception.MatrixElementAccessException;
import matrix.exception.MatrixMultiplicationDimensionException;
import matrix.exception.MatrixRowAccessException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

abstract sealed class MatrixBase implements Matrix permits MatrixBaseWithOperations {
    protected final DoubleBuffer buffer;

    public MatrixBase(DoubleBuffer buffer) {
        this.buffer = buffer;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj instanceof MatrixBase matrix) {
            return buffer.equals(matrix.buffer);
        }
        return false;
    }

    protected static void assertDimensionsEqual(Matrix A, Matrix B) {
        if (A.rows() != B.rows() || A.columns() != B.columns()) {
            throw new MatrixDimensionMismatchException(A, B);
        }
    }

    protected static void assertDimensionsEqual(Matrix X, int rows, int columns) {
        if (X.rows() != rows || X.columns() != columns) {
            throw new MatrixDimensionMismatchException(X, rows, columns);
        }
    }

    protected static void assertMultipliable(Matrix A, Matrix B) {
        if (A.columns() != B.rows()) {
            throw new MatrixMultiplicationDimensionException(A, B);
        }
    }

    static DoubleBuffer allocateDoubleBuffer(int capacity) {
        return ByteBuffer.allocateDirect(capacity * 8).order(ByteOrder.nativeOrder()).asDoubleBuffer();
    }

    static DoubleBuffer copyDoubleBuffer(DoubleBuffer data){
        return MatrixBase.copyDoubleBufferTo(data, MatrixBase.allocateDoubleBuffer(data.capacity()));
    }

    static DoubleBuffer copyDoubleBufferTo(DoubleBuffer data, DoubleBuffer out){
        data.rewind();
        out.put(data);
        return out;
    }

    protected static DoubleBuffer getBufferOf(Matrix m) {
        return ((MatrixBaseWithOperations) m).buffer;
    }

    protected void checkAccess(int row, int column) {
        if(row < 0 || column < 0 || row >= rows() || column > columns()) {
            throw new MatrixElementAccessException(this, row ,column);
        }
    }

    protected void checkRowAccess(int row) {
        if(row < 0 || row >= rows()) {
            throw new MatrixRowAccessException(this, row);
        }
    }

    protected void checkColumnAccess(int column) {
        if(column < 0 || column >= columns()) {
            throw new MatrixRowAccessException(this, column);
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder
                .append("Matrix<")
                .append(rows())
                .append(", ")
                .append(columns())
                .append(">[\n");
        for (int i = 0; i < rows(); i++) {
            builder.append("    ");
            for (int j = 0; j < columns(); j++) {
                builder.append("%.3f".formatted(getElementAt(i, j)));
                builder.append("\t");
            }
            builder.append("\n");
        }
        builder.append("]");
        return builder.toString();
    }

    @Override
    public final int size() {
        return rows() * columns();
    }
}
