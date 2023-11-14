package matrix;

import java.nio.DoubleBuffer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

abstract sealed class MatrixBaseWithOperations extends MatrixBase permits RectangularMatrix, VectorBase {
    protected MatrixBaseWithOperations(DoubleBuffer buffer) {
        super(buffer);
    }

    @Override
    public Matrix transpose() {
        return transpose_to(MatrixFactory.matrix(columns(), rows()));
    }

    @Override
    public Matrix transpose_to(Matrix out) {
        assertDimensionsEqual(out, columns(), rows());
        MatrixOperations.transpose(getBufferOf(out), buffer, rows(), columns());
        return out;
    }

    @Override
    public Matrix times(Matrix other) {
        assertMultipliable(this, other);

        int rows = this.rows();
        int columns = other.columns();

        return multiply_to(other, MatrixFactory.matrix(rows, columns));
    }

    @Override
    public Matrix multiply_to(Matrix other, Matrix out) {
        assertMultipliable(this, other);

        int rows = this.rows();
        int innerDimension = this.columns();
        int columns = other.columns();

        assertDimensionsEqual(out, rows, columns);

        MatrixOperations.multiply(
            getBufferOf(out),
            buffer,
            getBufferOf(other),
            rows, columns, innerDimension
        );
        return out;
    }

    @Override
    public Matrix multiply_and_add_to(Matrix other, Matrix out) {
        assertMultipliable(this, other);

        int rows = this.rows();
        int innerDimension = this.columns();
        int columns = other.columns();

        assertDimensionsEqual(out, rows, columns);

        MatrixOperations.multiplyAndAdd(
                getBufferOf(out),
                buffer,
                getBufferOf(other),
                rows, columns, innerDimension
        );
        return out;
    }

    @Override
    public Matrix times(double b) {
        return multiply_to(b, MatrixFactory.matrix(rows(), columns()));
    }

    @Override
    public Matrix multiply_to(double b, Matrix out) {
        assertDimensionsEqual(this, out);
        MatrixOperations.multiply(getBufferOf(out), buffer, b, rows(), columns());
        return out;
    }

    @Override
    public Matrix plus(Matrix other) {
        assertDimensionsEqual(this, other);
        return sum_to(other, MatrixFactory.matrix(rows(), columns()));
    }

    @Override
    public Matrix sum_to(Matrix other, Matrix out) {
        assertDimensionsEqual(this, other);
        assertDimensionsEqual(this, out);
        MatrixOperations.add(getBufferOf(out), buffer, getBufferOf(other), rows(), columns());
        return out;
    }

    @Override
    public Matrix plus(double b) {
        return sum_to(b, MatrixFactory.matrix(rows(), columns()));
    }

    @Override
    public Matrix sum_to(double b, Matrix out) {
        assertDimensionsEqual(this, out);
        MatrixOperations.add(getBufferOf(out), buffer, b, rows(), columns());
        return out;
    }

    @Override
    public Matrix minus(Matrix other) {
        assertDimensionsEqual(this, other);
        return subtract_to(other, MatrixFactory.matrix(rows(), columns()));
    }

    @Override
    public Matrix subtract_to(Matrix other, Matrix out) {
        assertDimensionsEqual(this, other);
        assertDimensionsEqual(this, out);
        MatrixOperations.subtract(getBufferOf(out), buffer, getBufferOf(other), rows(), columns());
        return out;
    }

    @Override
    public Matrix times_elementwise(Matrix other) {
        assertDimensionsEqual(this, other);
        return multiply_elementwise_to(other, MatrixFactory.matrix(rows(), columns()));
    }

    @Override
    public Matrix multiply_elementwise_to(Matrix other, Matrix out) {
        assertDimensionsEqual(this, other);
        assertDimensionsEqual(this, out);

        MatrixOperations.multiply(getBufferOf(out), buffer, getBufferOf(other), rows(), columns());
        return out;
    }

    @Override
    public double sumOfElements() {
        double total = 0;
        for(int i = 0; i < size(); i++) {
            total += buffer.get(i);
        }
        return total;
    }
}
