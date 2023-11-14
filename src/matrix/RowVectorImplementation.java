package matrix;

import matrix.exception.MatrixDimensionException;

import java.nio.DoubleBuffer;

final class RowVectorImplementation extends VectorBase implements RowVector{
    private final int size;

    public RowVectorImplementation(int size) {
        this(allocateDoubleBuffer(size), size);
    }

    public RowVectorImplementation(DoubleBuffer buffer, int size) {
        super(buffer);
        assert(size > 1);
        this.size = size;
    }

    @Override
    public int columns() {
        return size;
    }

    @Override
    public double getElementAt(int row, int column) {
        checkAccess(row, column);
        return getElementAt(column);
    }

    @Override
    public void setElementAt(int row, int column, double value) {
        checkAccess(row, column);
        setElementAt(column, value);
    }

    @Override
    public RowVector copy() {
        return new RowVectorImplementation(copyDoubleBuffer(buffer), size);
    }

    @Override
    public ColumnVector getColumn(int column) {
        checkRowAccess(column);
        return getValueMatrixAt(column);
    }

    @Override
    public RowVector getRow(int row) {
        checkColumnAccess(row);
        return this;
    }

    @Override
    public ColumnVector transpose() {
        return new ColumnVectorImplementation(buffer, size);
    }

    @Override
    public Matrix times(Matrix other) {
        if(other instanceof RectangularMatrix) {
            return multiply_to(other, new RowVectorImplementation(other.columns()));
        }
        else if(other instanceof ColumnVector) {
            return new DegenerateMatrix(dot((Vector)other));
        }
        return null;
    }

    @Override
    public Matrix multiply_to(Matrix other, Matrix out) {
        if(other.rows() != size()) {
            // TODO: elaborate
            throw new MatrixDimensionException("");
        }
        assertDimensionsEqual(out, 1, other.columns());
        MatrixOperations.multiply(getBufferOf(out), buffer, getBufferOf(other), 1, other.columns());
        return out;
    }
}
