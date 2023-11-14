package matrix;

import java.nio.DoubleBuffer;

final class MatrixOperations {
    private MatrixOperations() {
    }

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
        System.loadLibrary("matrix");
    }
}
