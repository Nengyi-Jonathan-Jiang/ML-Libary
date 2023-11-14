package matrix;

import matrix.exception.MatrixDimensionMismatchException;

import java.nio.DoubleBuffer;

final class DegenerateMatrix extends VectorBase implements Matrix, RowVector, ColumnVector {
    public DegenerateMatrix(double value) {
        this(allocateDoubleBuffer(1));
        setValue(value);
    }
    public DegenerateMatrix(DoubleBuffer buffer) {
        super(buffer);
    }

    @Override
    public double getElementAt(int row, int column) {
        checkAccess(row, column);
        return getElementAt(0);
    }

    public double getValue() {
        return getElementAt(0);
    }

    @Override
    public void setElementAt(int row, int column, double value) {
        checkAccess(row, column);
        setElementAt(0, value);
    }

    public void setValue(double value) {
        setElementAt(0, value);
    }

    @Override
    public DegenerateMatrix copy() {
        return new DegenerateMatrix(copyDoubleBuffer(buffer));
    }

    @Override
    public RowVector getRow(int row) {
        return this;
    }

    @Override
    public ColumnVector getColumn(int column) {
        return this;
    }

    @Override
    public DegenerateMatrix transpose() {
        return this;
    }

    @Override
    public Matrix times(Matrix other) {
        return other.times(getValue());
    }

    @Override
    public Matrix multiply_to(Matrix other, Matrix out) {
        return other.multiply_to(getValue(), out);
    }

    @Override
    public Matrix times(double b) {
        return new DegenerateMatrix(getValue() * b);
    }

    @Override
    public Matrix multiply_to(double b, Matrix out) {
        if(out instanceof DegenerateMatrix out_) {
            out_.setValue(getValue() * b);
            return out_;
        }
        throw new MatrixDimensionMismatchException(out, 1, 1);
    }

    @Override
    public Matrix sum_to(Matrix other, Matrix out) {
        return sum_to(((DegenerateMatrix)other).getValue(), out);
    }

    @Override
    public Matrix sum_to(double b, Matrix out) {
        assertDimensionsEqual(this, out);
        ((DegenerateMatrix)out).setValue(getValue() + b);
        return out;
    }

    @Override
    public Matrix subtract_to(Matrix other, Matrix out) {
        return sum_to(-((DegenerateMatrix)other).getValue(), out);
    }
}
