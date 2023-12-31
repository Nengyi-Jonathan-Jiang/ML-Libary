import matrix.Matrix;
import matrix.MatrixFactory;
import neuralnet.util.MathUtil;
import org.junit.Test;

import java.util.Arrays;

public class Benchmarker {
    @Test
    public void benchmark_mat_mul_mini(){
        benchmark_mat_mul(2, 10000000);
    }

    // RESULT: native outperforms java by up to 50% for 3x3 and above

    @Test
    public void benchmark_mat_mul_tiny(){
        benchmark_mat_mul(3, 1000000);
    }

    @Test
    public void benchmark_mat_mul_small(){
        benchmark_mat_mul(5, 100000);
    }

    @Test
    public void benchmark_mat_mul_large(){
        benchmark_mat_mul(100, 100);
    }

    @Test
    public void benchmark_mat_mul_huge(){
        benchmark_mat_mul(1000, 1);
    }


    public void benchmark_mat_mul(int SIZE, int ITERATIONS){
        double[] data = new double[SIZE * SIZE];
        double[] out = new double[SIZE * SIZE];
        Matrix mat = MatrixFactory.matrix(SIZE, SIZE);
        Matrix outMat = MatrixFactory.matrix(SIZE, SIZE);

        for(int n = 0; n < ITERATIONS; n++) {
            for(int i = 0; i < SIZE; i++) {
                for(int j = 0; j < SIZE; j++) {
                    double value = MathUtil.randGaussian();
                    data[i * SIZE + j] = value;
                    mat.setElementAt(i, j, value);
                }
            }

            mat.multiply_to(mat, outMat);

            matmul(data, out, SIZE);
        }
    }

    @Test
    public void benchmark_mat_mul_java(){
        int SIZE = 1000; int ITERATIONS = 1;
        double[] data = new double[SIZE * SIZE];
        double[] out = new double[SIZE * SIZE];
        for(int n = 0; n < ITERATIONS; n++) {
            for(int i = 0; i < SIZE; i++) {
                for(int j = 0; j < SIZE; j++) {
                    double value = MathUtil.randGaussian();
                    data[i * SIZE + j] = value;
                }
            }
            matmul(data, out, SIZE);
        }
    }

    @Test
    public void benchmark_mat_mul_native(){
        int SIZE = 1000; int ITERATIONS = 1;
        Matrix mat = MatrixFactory.matrix(SIZE, SIZE);
        Matrix outMat = MatrixFactory.matrix(SIZE, SIZE);
        for(int n = 0; n < ITERATIONS; n++) {
            for(int i = 0; i < SIZE; i++) {
                for(int j = 0; j < SIZE; j++) {
                    double value = MathUtil.randGaussian();
                    mat.setElementAt(i, j, value);
                }
            }
            mat.multiply_to(mat, outMat);
        }
    }

    public void matmul(double[] data, double[] out, int size) {
        Arrays.fill(out, 0);
        for(int i = 0; i < size; i++) {
            for(int k = 0; k < size; k++) {
                for (int j = 0; j < size; j++) {
                    out[i * size + j] += data[i * size + k] * data[k * size + j];
                }
            }
        }
    }
}
