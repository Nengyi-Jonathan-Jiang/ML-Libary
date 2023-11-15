package matrix;

import java.nio.DoubleBuffer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import static matrix.MatrixBase.*;

/**
 * Represents a matrix class in row major order
 */
public sealed interface Matrix permits DegenerateMatrix, MatrixBase, Vector {
    int rows();
    int columns();
    /** Total number of elements in the matrix, equal to rows() * columns() */
    int size();

    double getElementAt(int row, int column);
    void setElementAt(int row, int column, double value);

    Matrix copy();

    RowVector getRow(int row);
    ColumnVector getColumn(int column);

    Matrix transpose();
    Matrix transpose_to(Matrix out);
    
    Matrix times(Matrix other);
    Matrix multiply_to(Matrix other, Matrix out);
    Matrix multiply_and_add_to(Matrix other, Matrix out);
    Matrix times_elementwise(Matrix other);
    Matrix multiply_elementwise_to(Matrix other, Matrix out);
    Matrix times(double b);
    Matrix multiply_to(double b, Matrix out);
    Matrix plus(Matrix other);
    Matrix sum_to(Matrix other, Matrix out);
    Matrix plus(double b);
    Matrix sum_to(double b, Matrix out);
    Matrix minus(Matrix other);
    Matrix subtract_to(Matrix other, Matrix out);
    double sumOfElements();

    static Matrix transpose(Matrix m) {
        return m.transpose();
    }

    static Matrix transpose_to(Matrix m, Matrix out) {
        return m.transpose_to(out);
    }

    static Matrix multiply(Matrix a, Matrix b) {
        return a.times(b);
    }

    static Matrix multiply_to(Matrix a, Matrix b, Matrix out) {
        return a.multiply_to(b, out);
    }








    static Matrix applyOperation(Matrix A, DoubleUnaryOperator operation) {
        DoubleBuffer res = allocateDoubleBuffer(getBufferOf(A).capacity());

        for (int i = 0; i < res.capacity(); i++)
            res.put(i, operation.applyAsDouble(getBufferOf(A).get(i)));

        return MatrixFactory.matrix(res, A.rows(), A.columns());
    }

    static Matrix applyOperation(Matrix A, DoubleUnaryOperator operation, Matrix out) {
        assertDimensionsEqual(A, out);

        int totalElements = A.size();

        for (int i = 0; i < totalElements; i++)
            getBufferOf(out).put(i, operation.applyAsDouble(getBufferOf(A).get(i)));

        return out;
    }

    static Matrix applyOperation(Matrix A, Matrix B, DoubleBinaryOperator operation) {
        assertDimensionsEqual(A, B);

        DoubleBuffer res = allocateDoubleBuffer(A.size());

        for (int i = 0; i < res.capacity(); i++)
            res.put(i, operation.applyAsDouble(getBufferOf(A).get(i), getBufferOf(B).get(i)));

        return MatrixFactory.matrix(res, A.rows(), A.columns());
    }

    static Matrix applyOperation(Matrix A, Matrix B, DoubleBinaryOperator operation, Matrix out) {
        assertDimensionsEqual(A, B);
        assertDimensionsEqual(A, out);

        for (int i = 0; i < A.size(); i++)
            getBufferOf(out).put(i, operation.applyAsDouble(getBufferOf(A).get(i), getBufferOf(B).get(i)));

        return out;
    }

    static double applyOperationAndSum(Matrix A, Matrix B, DoubleBinaryOperator operation) {
        assertDimensionsEqual(A, B);

        double res = 0;

        for (int i = 0; i < A.size(); i++)
            res += operation.applyAsDouble(getBufferOf(A).get(i), getBufferOf(B).get(i));

        return res;
    }
}
