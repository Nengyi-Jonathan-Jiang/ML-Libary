package matrix;

import neuralnet.util.MathUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;

/**
 * Represents an immutable matrix class in row major order
 */
public class Matrix {
    /**
     * A buffer storing the data in column major order
     */
    private final DoubleBuffer data;
    /**
     * Number of rows in the matrix
     */
    public final int rows;
    /**
     * Number of columns in the matrix
     */
    public final int columns;


    public static Matrix identity(int size) {
        DoubleBuffer buffer = allocateDoubleBuffer(size, size);

        for (int i = 0; i < size; i++) {
            buffer.put(i * (size + 1), 1);
        }

        return new Matrix(buffer, size, size);
    }

    private static Matrix createMatrix(int rows, int columns, DoubleSupplier create) {
        DoubleBuffer buffer = allocateDoubleBuffer(rows, columns);

        for(int i = 0; i < rows * columns; i++) buffer.put(i, create.getAsDouble());

        return new Matrix(buffer, rows, columns);
    }

    public static Matrix zero(int rows, int columns) {
        return createMatrix(rows, columns, () -> 0);
    }

    public static Matrix one(int rows, int columns) {
        return createMatrix(rows, columns, () -> 1);
    }

    public static Matrix randomUniform(int rows, int columns) {
        return createMatrix(rows, columns, MathUtil::randUniform);
    }

    public static Matrix randomGaussian(int rows, int columns) {
        return createMatrix(rows, columns, MathUtil::randGaussian);
    }

    /**
     * Constructs a matrix from a 2d array of numbers in row-major order <br>
     * {@code data} must be rectangular
     */
    public Matrix(double[][] data) {
        int rows = data.length;
        if (rows == 0) throw new MatrixDimensionException("Invalid matrix dimensions: <0 , 0>");

        int columns = data[0].length;
        if (columns == 0) throw new MatrixDimensionException("Invalid matrix dimensions: <0, " + columns + ">");

        if (!Arrays.stream(data).allMatch(i -> i.length == columns))
            throw new MatrixDimensionException("Data must be a rectangular matrix");

        this.data = allocateDoubleBuffer(rows, columns);
        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                this.data.put(j + i * columns, data[i][j]);

        this.columns = columns;
        this.rows = rows;
    }

    /**
     * Constructs a matrix from raw data in column major order <br>
     */
    private Matrix(DoubleBuffer buf, int rows, int columns) {
        if (rows * columns == 0)
            throw new MatrixDimensionException("Invalid matrix dimensions: <" + rows + ", " + columns + ">");
        if (buf.capacity() != rows * columns) throw new MatrixDimensionException("Matrix dimensions do not match data");
        this.data = buf;
        this.rows = rows;
        this.columns = columns;
    }

    public double at(int row, int column) {
        return data.get(column + row * columns);
    }

    public Matrix getRow(int row) {
        DoubleBuffer columnData = data.slice(row * columns, columns);
        return new Matrix(columnData, 1, columns);
    }

    /**
     * This is a bit slower than getRow since data is stored in column major order
     */
    public Matrix getColumn(int column) {
        DoubleBuffer res = allocateDoubleBuffer(rows);
        for (int r = 0; r < rows; r++) res.put(r, at(r, column));
        return new Matrix(res, rows, 1);
    }

    public boolean isColumnVector() {
        return columns == 1;
    }

    public boolean isRowVector() {
        return rows == 1;
    }

    /**
     * Finds the matrix transpose of {@code X}
     */
    public static Matrix transpose(Matrix X) {
        DoubleBuffer out = allocateDoubleBuffer(X.columns, X.rows);
        transpose(out, X.data, X.columns, X.rows);
        return new Matrix(out, X.columns, X.rows);
    }

    public static Matrix multiply(Matrix A, Matrix B) {
        if(A.columns != B.rows) {
            throw new MatrixDimensionException(
                    "Cannot multiply matrices <%d, %d> and <%d, %d>"
                    .formatted(A.rows, A.columns, B.rows, B.columns)
            );
        }

        int outputRows = A.rows, outputColumns = B.columns, innerDimension = A.columns;
        
        DoubleBuffer out = allocateDoubleBuffer(outputRows, outputColumns);
        multiply(out, A.data, B.data, outputRows, outputColumns, innerDimension);
        return new Matrix(out, outputRows, outputColumns);
    }

    public static Matrix multiply_elementWise(Matrix A, Matrix B) {
        checkDimensionsEqual(A, B);

        int rows = A.rows, columns = A.columns;
        DoubleBuffer out = allocateDoubleBuffer(rows, columns);
        multiply(out, A.data, B.data, rows, columns);
        return new Matrix(out, rows, columns);
    }

    public static Matrix multiply(Matrix X, double c) {
        int rows = X.rows, columns = X.columns;
        DoubleBuffer out = allocateDoubleBuffer(rows, columns);
        multiply(out, X.data, c, rows, columns);
        return new Matrix(out, rows, columns);
    }
    
    public static Matrix add(Matrix A, Matrix B) {
        checkDimensionsEqual(A, B);

        int rows = A.rows, columns = A.columns;
        DoubleBuffer out = allocateDoubleBuffer(rows, columns);
        add(out, A.data, B.data, rows, columns);
        return new Matrix(out, rows, columns);
    }

    public static Matrix add(Matrix X, double c) {
        int rows = X.rows, columns = X.columns;
        DoubleBuffer out = allocateDoubleBuffer(rows, columns);
        add(out, X.data, c, rows, columns);
        return new Matrix(out, rows, columns);
    }

    public static Matrix subtract(Matrix A, Matrix B) {
        checkDimensionsEqual(A, B);

        int rows = A.rows, columns = A.columns;
        DoubleBuffer out = allocateDoubleBuffer(rows, columns);
        subtract(out, A.data, B.data, rows, columns);
        return new Matrix(out, rows, columns);
    }

    public Matrix transpose() {
        return transpose(this);
    }

    public Matrix times(Matrix other) {
        return multiply(this, other);
    }

    public Matrix times_elementwise(Matrix other) {
        return multiply_elementWise(this, other);
    }

    public Matrix times(double b) {
        return multiply(this, b);
    }

    public Matrix plus(Matrix other) {
        return add(this, other);
    }

    public Matrix plus(double b) {
        return add(this, b);
    }

    public Matrix minus(Matrix other) {
        return subtract(this, other);
    }

    public static Matrix applyOperation(Matrix A, DoubleUnaryOperator operation) {
        DoubleBuffer res = allocateDoubleBuffer(A.data.capacity());

        for (int i = 0; i < res.capacity(); i++)
            res.put(i, operation.applyAsDouble(A.data.get(i)));

        return new Matrix(res, A.rows, A.columns);
    }

    public static Matrix applyOperation(Matrix A, Matrix B, DoubleBinaryOperator operation) {
        checkDimensionsEqual(A, B);

        DoubleBuffer res = allocateDoubleBuffer(A.data.capacity());

        for (int i = 0; i < res.capacity(); i++)
            res.put(i, operation.applyAsDouble(A.data.get(i), B.data.get(i)));

        return new Matrix(res, A.rows, A.columns);
    }

    private static void checkDimensionsEqual(Matrix A, Matrix B) {
        if (A.rows != B.rows || A.columns != B.columns)
            throw new MatrixDimensionException("Dimension mismatch: <" + A.rows + ", " + A.columns + "> != <" + B.rows + ", " + B.columns + ">");
    }

    @Override
    public boolean equals(Object obj) {
        return equals(obj, 0.01);
    }

    public boolean equals(Object obj, double tolerance) {
        DoubleBuffer data;
        if (obj instanceof Matrix) {
            data = ((Matrix) obj).data;
            checkDimensionsEqual(this, (Matrix) obj);
        } else if (obj instanceof double[]) {
            data = DoubleBuffer.allocate(((double[]) obj).length);
            data.put((double[]) obj);
        } else return false;

        if (data.capacity() != columns * rows) return false;

        if (tolerance == 0) return this.data.equals(data);

        for (int i = 0; i < columns * rows; i++) {
            if (Math.abs(this.data.get(i) - data.get(i)) > tolerance) {
                return false;
            }
        }
        return true;
    }

    private String _toString() {
        return "Matrix<" + rows + ", " + columns + ">";
    }

    @Override
    public String toString() {
        return toString(true);
    }

    public String toString(boolean printData) {
        if (!printData) return _toString();
        StringBuilder builder = new StringBuilder();
        builder.append(_toString());
        builder.append("[\n");
        for (int i = 0; i < rows; i++) {
            builder.append("    ");
            for (int j = 0; j < columns; j++) {
                builder.append("%.3f".formatted(at(i, j)));
                builder.append("\t");
            }
            builder.append("\n");
        }
        builder.append("]");
        return builder.toString();
    }

    private static DoubleBuffer allocateDoubleBuffer(int rows, int columns) {
        return allocateDoubleBuffer(rows * columns);
    }

    private static DoubleBuffer allocateDoubleBuffer(int capacity) {
        return ByteBuffer.allocateDirect(capacity * 8).order(ByteOrder.nativeOrder()).asDoubleBuffer();
    }

    private static native void transpose(DoubleBuffer out, DoubleBuffer X, int rows, int columns);

    private static native void multiply(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns, int innerDimension);

    private static native void multiply(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    private static native void multiply(DoubleBuffer out, DoubleBuffer X, double c, double rows, double columns);

    private static native void add(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    private static native void add(DoubleBuffer out, DoubleBuffer X, double c, int rows, int columns);

    private static native void subtract(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    static {
        System.loadLibrary("matrix");
    }
}
