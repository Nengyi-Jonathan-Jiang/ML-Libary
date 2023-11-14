package matrix;

import matrix.exception.MatrixDimensionException;
import matrix.exception.MatrixElementAccessException;
import matrix.exception.MatrixException;

import java.nio.DoubleBuffer;

abstract sealed class VectorBase extends MatrixBaseWithOperations implements Vector permits ColumnVectorImplementation, RowVectorImplementation, DegenerateMatrix {
    protected VectorBase(DoubleBuffer buffer) {
        super(buffer);
    }

    protected DegenerateMatrix getValueMatrixAt(int index) {
        checkAccess(index);
        return new DegenerateMatrix(buffer.slice(index, 1));
    }

    @Override
    public final void setElementAt(int index, double value) {
        checkAccess(index);
        buffer.put(index, value);
    }

    @Override
    public final double getElementAt(int index) {
        checkAccess(index);
        return buffer.get(index);
    }

    private void checkAccess(int index) {
        if(index < 0 || index >= size()) {
            throw new MatrixElementAccessException(this, index, index);
        }
    }

    @Override
    public Matrix transpose_to(Matrix out) {
        if(out instanceof Vector out_) {
            return transpose_to(out_);
        }
        else throw new MatrixDimensionException("");
    }

    @Override
    public Matrix times(double b) {
        return multiply_to(b, MatrixFactory.matrix(allocateDoubleBuffer(size()), rows(), columns()));
    }

    @Override
    public Matrix multiply_to(double b, Matrix out) {
        assertDimensionsEqual(this, out);
        MatrixOperations.multiply(getBufferOf(out), buffer, b, rows(), columns());
        return out;
    }

    @Override
    public Matrix transpose() {
        return MatrixFactory.matrix(buffer, columns(), rows());
    }

    @Override
    public double dot(Vector other) {
        if(this instanceof DegenerateMatrix A && other instanceof DegenerateMatrix B) {
            return A.getValue() * B.getValue();
        }
        else if (other.size() == size()) {
            return MatrixOperations.dot(buffer, getBufferOf(other), size());
        }
        // TODO: make more specific
        throw new MatrixException();
    }

    @Override
    public Matrix outer(Vector other) {
        return outer_to(other, MatrixFactory.matrix(size(), other.size()));
    }

    @Override
    public Matrix outer_to(Vector other, Matrix out) {
        assertDimensionsEqual(out, size(), other.size());
        MatrixOperations.outer(getBufferOf(out), buffer, getBufferOf(other), size(), other.size());
        return out;
    }
}
