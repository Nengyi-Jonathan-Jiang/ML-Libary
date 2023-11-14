package matrix;

import java.nio.DoubleBuffer;

final class ColumnVectorImplementation extends VectorBase implements ColumnVector {
    private final int size;

    public ColumnVectorImplementation(int size) {
        this(allocateDoubleBuffer(size), size);
    }

    public ColumnVectorImplementation(DoubleBuffer buffer, int size) {
        super(buffer);
        assert(size > 1);
        this.size = size;
    }

    @Override
    public int rows() {
        return size;
    }

    @Override
    public double getElementAt(int row, int column) {
        checkAccess(row, column);
        return getElementAt(row);
    }

    @Override
    public void setElementAt(int row, int column, double value) {
        checkAccess(row, column);
        setElementAt(row, value);
    }

    @Override
    public ColumnVector copy() {
        return new ColumnVectorImplementation(copyDoubleBuffer(buffer), size);
    }

    @Override
    public ColumnVector getColumn(int column) {
        checkColumnAccess(column);
        return this;
    }

    @Override
    public RowVector getRow(int row) {
        checkRowAccess(row);
        return getValueMatrixAt(row);
    }

    @Override
    public RowVector transpose() {
        return new RowVectorImplementation(buffer, size);
    }

    // TODO: add faster multiplication
}
