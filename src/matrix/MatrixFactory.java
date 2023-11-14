package matrix;

import matrix.exception.MatrixDimensionException;
import neuralnet.util.MathUtil;

import java.nio.DoubleBuffer;
import java.util.function.DoubleSupplier;

public class MatrixFactory {
    public static Matrix matrix(double[][] data) {
        int rows = data.length;
        if (rows == 0) throw new MatrixDimensionException("Invalid matrix dimensions: <0 , 0>");

        int columns = data[0].length;
        if (columns == 0) throw new MatrixDimensionException("Invalid matrix dimensions: <%d, 0>".formatted(rows));

        for (double[] row : data) {
            if (row.length != columns) {
                throw new MatrixDimensionException("Data must be a rectangular matrix");
            }
        }

        DoubleBuffer buffer = MatrixBase.allocateDoubleBuffer(rows * columns);
        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                buffer.put(j + i * columns, data[i][j]);

        return matrix(buffer, rows, columns);
    }

    /** Create a matrix with the same size as other */
    public static Matrix matrix(Matrix other) {
        return matrix(other.rows(), other.columns());
    }

    public static Matrix matrix(int rows, int columns) {
        return matrix(MatrixBase.allocateDoubleBuffer(rows * columns), rows, columns);
    }

    public static Matrix matrix(DoubleBuffer buffer, int rows, int columns) {
        if(rows == 1 && columns == 1)
            return new DegenerateMatrix(buffer);
        else if (rows == 1)
            return new RowVectorImplementation(buffer, columns);
        else if (columns == 1)
            return new ColumnVectorImplementation(buffer, rows);
        else
            return new RectangularMatrix(buffer, rows, columns);
    }

    public static RowVector rowVector(int size) {
        return (RowVector) matrix(1, size);
    }

    public static RowVector rowVector(DoubleBuffer buffer, int size) {
        return (RowVector) matrix(buffer, 1, size);
    }

    public static ColumnVector columnVector(int size) {
        return (ColumnVector) matrix(size, 1);
    }

    public static ColumnVector columnVector(DoubleBuffer buffer, int size) {
        return (ColumnVector) matrix(buffer, size, 1);
    }

    public static RowVector rowVector(double... data) {
        int size = data.length;
        if (size == 0) throw new MatrixDimensionException("Invalid matrix dimensions: <1, 0>");

        DoubleBuffer buffer = MatrixBase.allocateDoubleBuffer(size);
        buffer.put(data);

        return rowVector(buffer, size);
    }

    public static ColumnVector columnVector(double... data) {
        int size = data.length;
        if (size == 0) throw new MatrixDimensionException("Invalid matrix dimensions: <0, 1>");

        DoubleBuffer buffer = MatrixBase.allocateDoubleBuffer(size);
        buffer.put(data);

        return columnVector(buffer, size);
    }


    public static Matrix identity(int size) {
        Matrix res = matrix(size, size);
        for (int i = 0; i < size; i++) {
            res.setElementAt(i, i, 1);
        }
        return res;
    }
    public static Matrix matrix(int rows, int columns, DoubleSupplier create) {
        DoubleBuffer buffer = MatrixBase.allocateDoubleBuffer(rows * columns);

        for (int i = 0; i < rows * columns; i++) buffer.put(i, create.getAsDouble());

        return matrix(buffer, rows, columns);
    }

    public static Matrix zero(int rows, int columns) {
        return matrix(rows, columns);
    }

    public static Matrix ones(int rows, int columns) {
        return matrix(rows, columns, () -> 1);
    }

    public static Matrix randomUniform(int rows, int columns) {
        return matrix(rows, columns, MathUtil::randUniform);
    }

    public static Matrix randomGaussian(int rows, int columns) {
        return matrix(rows, columns, MathUtil::randGaussian);
    }
}
