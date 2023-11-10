package neuralnet.util;

import java.security.InvalidParameterException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public final class ArrayUtil {
    private ArrayUtil() {}

    public static void apply(double[] input, DoubleUnaryOperator operation) {
        for(int i = 0; i < input.length; i++) {
            input[i] = operation.applyAsDouble(input[i]);
        }
    }

    public static void apply(double[] arr, double[] data, DoubleBinaryOperator operation) {
        assertArrayLengthsEqual(arr, data);
        for(int i = 0; i < arr.length; i++) {
            arr[i] = operation.applyAsDouble(arr[i], data[i]);
        }
    }

    public static double[] map(double[] input, DoubleUnaryOperator operation) {
        double[] output = new double[input.length];
        for(int i = 0; i < input.length; i++) {
            output[i] = operation.applyAsDouble(input[i]);
        }

        return output;
    }

    public static void map(double[] input, DoubleUnaryOperator operation, double[] output) {
        assertArrayLengthsEqual(input, output);

        for(int i = 0; i < input.length; i++) {
            output[i] = operation.applyAsDouble(input[i]);
        }
    }

    public static double[] map(double[] a, double[] b, DoubleBinaryOperator operation) {
        assertArrayLengthsEqual(a, b);

        int inputLength = a.length;

        double[] output = new double[inputLength];
        for(int i = 0; i < inputLength; i++) {
            output[i] = operation.applyAsDouble(a[i], b[i]);
        }

        return output;
    }

    public static double[] multiplyElementwise(double[] a, double[] b) {
        assertArrayLengthsEqual(a, b);
        int inputLength = a.length;

        double[] output = new double[inputLength];
        for(int i = 0; i < inputLength; i++) {
            output[i] = a[i] * b[i];
        }

        return output;
    }

    public static double[] sumElementwise(double[] a, double[] b) {
        assertArrayLengthsEqual(a, b);
        int inputLength = a.length;

        double[] output = new double[inputLength];
        for(int i = 0; i < inputLength; i++) {
            output[i] = a[i] + b[i];
        }

        return output;
    }

    public static double[] multiplyByConstant(double[] a, double c) {
        double[] output = new double[a.length];
        for(int i = 0; i < a.length; i++) {
            output[i] = a[i] * c;
        }

        return output;
    }

    private static void assertArrayLengthsEqual(double[] a, double[] b) {
        if(a.length != b.length) {
            throw new InvalidParameterException("Array length mismatch: %s should equal %s".formatted(a.length, b.length));
        }
    }
}
