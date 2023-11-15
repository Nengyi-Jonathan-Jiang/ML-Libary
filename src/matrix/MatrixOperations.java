package matrix;

import java.nio.DoubleBuffer;

final class MatrixOperations {
    private static final boolean USE_NATIVE = true;

    private MatrixOperations() {}

    static void transpose_java(DoubleBuffer out, DoubleBuffer X, int rows, int columns) {
        for(int row = 0; row < rows; row++) {
            for(int column = 0; column < columns; column++) {
                int index = row * columns + column;
                out.put(index, X.get(row * columns + column));
            }
        }
    }

    static void multiply_java(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns, int innerDimension) {}

    static void multiplyAndAdd_java(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns, int innerDimension) {}

    static void multiply_java(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns) {}

    static void multiply_java(DoubleBuffer out, DoubleBuffer X, double c, double rows, double columns) {}

    static void add_java(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns) {}

    static void add_java(DoubleBuffer out, DoubleBuffer X, double c, int rows, int columns) {}

    static void subtract_java(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns) {}

    static double dot_java(DoubleBuffer A, DoubleBuffer B, int size) {
        return 0;
    }

    static void outer_java(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns) {}

    static native void transpose(DoubleBuffer out, DoubleBuffer X, int rows, int columns);

    static native void multiply(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns, int innerDimension);

    static native void multiplyAndAdd(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns, int innerDimension);

    static native void multiply(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    static native void multiply(DoubleBuffer out, DoubleBuffer X, double c, double rows, double columns);

    static native void add(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    static native void add(DoubleBuffer out, DoubleBuffer X, double c, int rows, int columns);

    static native void subtract(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    static native double dot(DoubleBuffer A, DoubleBuffer B, int size);

    static native void outer(DoubleBuffer out, DoubleBuffer A, DoubleBuffer B, int rows, int columns);

    static {
        if(USE_NATIVE) System.loadLibrary("matrix");
    }
}
